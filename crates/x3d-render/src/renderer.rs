//! The renderer: turns an [`x3d_scene::Scene`] into pixels.
//!
//! 🎨 **One frame, step by step** (see [`Renderer::render`])
//! 1. Write the per-frame uniform (camera matrices, lights).
//! 2. Order the draws: opaque first, then transparent back-to-front.
//! 3. Record one *render pass*: clear colour + depth, then for each draw
//!    set the pipeline (only when it changes), point group 1 at this draw's
//!    slice of the big uniform buffer, bind its vertex/index buffers, draw.
//! 4. Submit the command buffer to the queue. The GPU runs it later.
//!
//! 🎨 **Pipelines** bundle shaders + fixed-function state (topology, culling,
//! blending, depth test). They're expensive to create, so we build each
//! combination once, on first use, and cache it in a `HashMap`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use wgpu::util::DeviceExt;
use x3d_scene::runtime::Changes;
use x3d_scene::{Background, Draw, Light, LodTag, Mesh, Scene, SwitchGuard, Topology, Vertex};

use crate::camera::OrbitCamera;
use crate::gpu::Gpu;
use crate::textures::{MaterialTextures, TextureCache};
use crate::uniforms::{DrawUniform, FrameUniform, GpuLight, MAX_LIGHTS};

/// The shader compiled into the binary. `--shader path.wgsl` overrides it.
pub const BUILTIN_SHADER: &str = include_str!("shaders/scene.wgsl");
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// What the fragment shader outputs. Matches the `VIEW_*` constants in WGSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum DebugView {
    /// X3D lighting (the real picture).
    #[default]
    Lit,
    /// World-space normals as colours (x→red, y→green, z→blue).
    Normals,
    /// Distance from the camera (white = near).
    Depth,
    /// Material or vertex colour, before lighting.
    BaseColor,
    /// Lighting on a plain grey material.
    LightingOnly,
    /// Front faces green, back faces red: a winding-order detector.
    Facing,
    /// Texture coordinates as red (s) and green (t), repeating every unit.
    Uv,
}

impl DebugView {
    pub const ALL: [Self; 7] = [
        Self::Lit,
        Self::Normals,
        Self::Depth,
        Self::BaseColor,
        Self::LightingOnly,
        Self::Facing,
        Self::Uv,
    ];

    /// Cycle to the next view (wraps around).
    pub fn next(self) -> Self {
        Self::ALL[(self as usize + 1) % Self::ALL.len()]
    }
}

impl fmt::Display for DebugView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 🦀 `{:?}` of a fieldless enum prints its name; reuse that.
        write!(f, "{self:?}")
    }
}

/// How colours are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ColorMode {
    /// Light in display (sRGB) space, like X_ITE does for classic Materials.
    #[default]
    Display,
    /// Decode colours to linear, light, re-encode. Physically nicer falloff.
    Linear,
}

/// Knobs you can flip at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    pub view: DebugView,
    pub color_mode: ColorMode,
    /// Draw triangle edges only (needs `POLYGON_MODE_LINE`).
    pub wireframe: bool,
    /// `None`: follow the scene's NavigationInfo.
    pub headlight: Option<bool>,
    /// Cull back faces of `solid='true'` geometry (X3D's rule).
    pub cull_back_faces: bool,
    pub show_lines: bool,
    pub show_points: bool,
    /// Multiplies PhysicalMaterial's linear light before display (1 = as X_ITE).
    pub exposure: f32,
    /// Multiplies line widths and point sizes (e.g. 2 on a HiDPI screen).
    pub line_scale: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            view: DebugView::Lit,
            color_mode: ColorMode::Display,
            wireframe: false,
            headlight: None,
            cull_back_faces: true,
            show_lines: true,
            show_points: true,
            exposure: 1.0,
            line_scale: 1.0,
        }
    }
}

/// Numbers for the HUD.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameStats {
    pub draw_calls: u32,
    pub pipeline_switches: u32,
    pub lights: u32,
    /// CPU time to record and submit (not GPU time — that needs timestamp
    /// queries; see LEARNING.md).
    pub cpu_ms: f32,
}

/// Triangle meshes are vertex + index buffers. Lines and points are one
/// buffer of [`Segment`]s, drawn as instanced quads.
#[derive(Debug)]
enum GpuMesh {
    Triangles {
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        index_count: u32,
        solid: bool,
    },
    Wide {
        segments: wgpu::Buffer,
        count: u32,
        topology: Topology,
    },
}

impl GpuMesh {
    fn topology(&self) -> Topology {
        match self {
            Self::Triangles { .. } => Topology::Triangles,
            Self::Wide { topology, .. } => *topology,
        }
    }
    fn solid(&self) -> bool {
        matches!(self, Self::Triangles { solid: true, .. })
    }
    fn bytes(&self) -> u64 {
        match self {
            Self::Triangles {
                vertices, indices, ..
            } => vertices.size() + indices.size(),
            Self::Wide { segments, .. } => segments.size(),
        }
    }
}

/// One line segment (or one point: `a == b`), as the `vs_wide` shader reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Segment {
    a: [f32; 3],
    b: [f32; 3],
    color_a: [f32; 4],
    color_b: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
struct DrawItem {
    mesh: usize,
    /// Byte offset of this draw's `DrawUniform` in `draw_buffer`.
    offset: u32,
    /// Index into `Renderer::materials` (its texture bind group).
    material: usize,
    /// Index into `Renderer::lods`: the LOD levels this draw belongs to.
    lod: usize,
    /// Index into `Renderer::guards`: the animated Switches it sits under.
    guard: usize,
    /// The `Scene::draws` entry this came from (for animation updates).
    source: usize,
    texture_bits: u32,
    texture_alpha: bool,
    light_mask: u32,
    transparent: bool,
    center: Vec3,
}

/// Every distinct pipeline we might need.
/// 🦀 `#[derive(Hash, Eq)]` is all it takes to use a struct as a map key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    topology: Topology,
    blend: bool,
    cull: bool,
    wireframe: bool,
}

#[derive(Debug)]
struct Targets {
    size: (u32, u32),
    depth: wgpu::TextureView,
    /// The multisampled colour buffer (resolved into the real target).
    msaa: Option<wgpu::TextureView>,
}

#[derive(Debug)]
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    color_format: wgpu::TextureFormat,
    sample_count: u32,
    supports_wireframe: bool,

    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,

    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    draw_layout: wgpu::BindGroupLayout,
    draw_buffer: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
    /// Bytes between consecutive `DrawUniform`s (≥ 192, rounded up to the
    /// device's `min_uniform_buffer_offset_alignment`, usually 256).
    draw_stride: u32,

    meshes: Vec<GpuMesh>,
    draws: Vec<DrawItem>,
    textures: TextureCache,
    materials: Vec<Arc<MaterialTextures>>,
    /// Distinct LOD tag lists; index 0 is "not inside any LOD".
    lods: Vec<Arc<[LodTag]>>,
    /// Distinct Switch guard lists; index 0 is "under no animated Switch".
    guards: Vec<Arc<[SwitchGuard]>>,
    /// Current `whichChoice` of each animated Switch.
    switches: Vec<i32>,
    lights: Vec<Light>,
    background: Background,
    scene_headlight: bool,
    targets: Option<Targets>,
    warned_lights: bool,
}

impl Renderer {
    /// `color_format`: the format of the textures you will render into.
    /// `sample_count`: 1 (no MSAA) or 4 (smooth edges).
    pub fn new(gpu: &Gpu, color_format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let device = gpu.device.clone(); // 🦀 wgpu handles are cheap, reference-counted clones
        let sample_count = supported_samples(gpu, color_format, sample_count);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene.wgsl (built in)"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BUILTIN_SHADER)),
        });

        let uniform_entry = |dynamic: bool, size: usize| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame layout"),
            entries: &[uniform_entry(false, size_of::<FrameUniform>())],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw layout"),
            entries: &[uniform_entry(true, size_of::<DrawUniform>())],
        });
        let textures = TextureCache::new(&device, &gpu.queue);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&draw_layout),
                Some(textures.layout()),
            ],
            immediate_size: 0,
        });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bind group"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        let align = device.limits().min_uniform_buffer_offset_alignment;
        let draw_stride = (size_of::<DrawUniform>() as u32).next_multiple_of(align);
        let (draw_buffer, draw_bind_group) =
            Self::draw_buffer(&device, &draw_layout, draw_stride, 1);

        Self {
            queue: gpu.queue.clone(),
            color_format,
            sample_count,
            supports_wireframe: gpu.supports_wireframe(),
            shader,
            pipeline_layout,
            pipelines: HashMap::new(),
            frame_buffer,
            frame_bind_group,
            draw_layout,
            draw_buffer,
            draw_bind_group,
            draw_stride,
            meshes: Vec::new(),
            draws: Vec::new(),
            textures,
            materials: Vec::new(),
            lods: Vec::new(),
            guards: Vec::new(),
            switches: Vec::new(),
            lights: Vec::new(),
            background: Background::default(),
            scene_headlight: true,
            targets: None,
            warned_lights: false,
            device,
        }
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }
    pub fn supports_wireframe(&self) -> bool {
        self.supports_wireframe
    }
    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    fn draw_buffer(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        count: usize,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw uniforms"),
            size: u64::from(stride) * count.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw bind group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // Bind a *window* of one DrawUniform; the dynamic offset slides it.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(size_of::<DrawUniform>() as u64),
                }),
            }],
        });
        (buffer, bind_group)
    }

    /// Upload a scene: vertex/index buffers for each unique mesh, and one
    /// big uniform buffer holding every draw's matrices and material.
    pub fn set_scene(&mut self, scene: &Scene) {
        self.meshes.clear();
        self.draws.clear();
        self.materials.clear();
        self.lods = vec![Arc::from([])];
        self.guards = vec![Arc::from([])];
        let mut material_of: HashMap<*const MaterialTextures, usize> = HashMap::new();

        // 🦀 `Arc::as_ptr` gives a stable identity for shared meshes, so a
        // DEF'd mesh USE'd 100 times is uploaded once.
        let mut index_of: HashMap<*const Mesh, usize> = HashMap::new();

        for (source, draw) in scene.draws.iter().enumerate() {
            let mesh = &draw.mesh;
            if mesh.indices.is_empty() {
                continue;
            }
            let mesh_index = *index_of.entry(Arc::as_ptr(mesh)).or_insert_with(|| {
                self.meshes.push(upload(&self.device, mesh));
                self.meshes.len() - 1
            });
            // Textures are cached across scenes, so a hot reload doesn't
            // decode every image again.
            let textures = self.textures.material(&draw.material.textures);
            let (texture_bits, texture_alpha) = (textures.bits, textures.has_alpha);
            let material = *material_of
                .entry(Arc::as_ptr(&textures))
                .or_insert_with(|| {
                    self.materials.push(textures);
                    self.materials.len() - 1
                });
            // 🦀 `fold` builds the bit mask: bit i = scene light i reaches this draw.
            let light_mask = scene
                .lights
                .iter()
                .take(32)
                .enumerate()
                .fold(0u32, |mask, (i, l)| {
                    if l.reaches(&draw.scopes) {
                        mask | 1 << i
                    } else {
                        mask
                    }
                });
            let lod = intern_lod(&mut self.lods, &draw.lod);
            let guard = intern_guard(&mut self.guards, &draw.switches);
            self.draws.push(DrawItem {
                mesh: mesh_index,
                offset: self.draw_stride * self.draws.len() as u32,
                material,
                lod,
                guard,
                source,
                texture_bits,
                texture_alpha,
                light_mask,
                transparent: false,
                center: Vec3::ZERO,
            });
        }
        if (self.draw_stride as usize * self.draws.len().max(1)) as u64 > self.draw_buffer.size() {
            (self.draw_buffer, self.draw_bind_group) = Self::draw_buffer(
                &self.device,
                &self.draw_layout,
                self.draw_stride,
                self.draws.len(),
            );
        }
        self.write_draw_uniforms(scene);

        self.lights.clone_from(&scene.lights);
        self.switches.clone_from(&scene.switches);
        self.background = scene.background.clone();
        self.scene_headlight = scene.headlight;
        log::info!(
            "uploaded {} meshes, {} draws, {} materials ({} KiB of vertices, {} KiB of textures)",
            self.meshes.len(),
            self.draws.len(),
            self.materials.len(),
            self.meshes.iter().map(GpuMesh::bytes).sum::<u64>() / 1024,
            self.textures.bytes() / 1024
        );
    }

    /// Follow an animated scene: call after [`Scene::advance`] with what it
    /// changed. Cheap when nothing did.
    pub fn update(&mut self, scene: &Scene, changes: &Changes) {
        if changes.transforms || changes.materials {
            self.write_draw_uniforms(scene);
        }
        if changes.lights {
            self.lights.clone_from(&scene.lights);
        }
        if changes.switches {
            self.switches.clone_from(&scene.switches);
        }
    }

    /// (Re)write every draw's matrices and material into the uniform buffer.
    fn write_draw_uniforms(&mut self, scene: &Scene) {
        let stride = self.draw_stride as usize;
        let mut bytes = vec![0u8; stride * self.draws.len().max(1)];
        for (item, slot) in self.draws.iter_mut().zip(bytes.chunks_exact_mut(stride)) {
            let draw: &Draw = &scene.draws[item.source];
            let uniform = DrawUniform::new(
                draw.transform,
                &draw.material,
                draw.mesh.has_colors,
                item.texture_bits,
                item.light_mask,
            );
            slot[..size_of::<DrawUniform>()].copy_from_slice(bytemuck::bytes_of(&uniform));
            item.transparent = draw.material.is_transparent() || item.texture_alpha;
            item.center = draw.world_bounds().center();
        }
        self.queue.write_buffer(&self.draw_buffer, 0, &bytes);
    }

    /// Replace the shader. On any compile/validation error the old shader
    /// keeps running and the error text comes back as `Err`.
    pub fn reload_shader(&mut self, source: &str) -> Result<(), String> {
        // 🎨 Error scopes catch validation errors instead of sending them to
        // the global handler, so we can decide what to do with them.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scene.wgsl (reloaded)"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
        // Build one pipeline too: that catches entry points or bindings that
        // no longer match the Rust side.
        let key = PipelineKey {
            topology: Topology::Triangles,
            blend: false,
            cull: true,
            wireframe: false,
        };
        let probe = self.create_pipeline(&module, key);
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(err.to_string());
        }
        self.shader = module;
        self.pipelines.clear();
        self.pipelines.insert(key, probe);
        Ok(())
    }

    fn create_pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        key: PipelineKey,
    ) -> wgpu::RenderPipeline {
        // Triangles read one Vertex per vertex; lines and points read one
        // Segment per *instance* and build a quad from it (see `vs_wide`).
        let wide = key.topology != Topology::Triangles;
        let vertex_layout = if wide {
            wgpu::VertexBufferLayout {
                array_stride: size_of::<Segment>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x4
                ],
            }
        } else {
            wgpu::VertexBufferLayout {
                array_stride: size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                // position, normal, colour, texcoord → @location(0) … (3) in the shader.
                attributes: &wgpu::vertex_attr_array![
                    0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x3
                ],
            }
        };
        let topology = wgpu::PrimitiveTopology::TriangleList;
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{key:?}")),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(if wide { "vs_wide" } else { "vs_main" }),
                    compilation_options: Default::default(),
                    buffers: &[Some(vertex_layout)],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    front_face: wgpu::FrontFace::Ccw, // X3D's ccw='true'
                    cull_mode: key.cull.then_some(wgpu::Face::Back),
                    polygon_mode: if key.wireframe {
                        wgpu::PolygonMode::Line
                    } else {
                        wgpu::PolygonMode::Fill
                    },
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Transparent surfaces test depth but don't write it, so
                    // things behind them still draw.
                    depth_write_enabled: Some(!key.blend),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: self.sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.color_format,
                        blend: key.blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn ensure_targets(&mut self, size: (u32, u32)) {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return;
        }
        let texture = |label, format, samples| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let depth = texture("depth", DEPTH_FORMAT, self.sample_count);
        let msaa = (self.sample_count > 1)
            .then(|| texture("msaa colour", self.color_format, self.sample_count));
        self.targets = Some(Targets { size, depth, msaa });
    }

    /// Draw one frame into `target` (a view of a `color_format` texture).
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        size: (u32, u32),
        camera: &OrbitCamera,
        settings: &RenderSettings,
    ) -> FrameStats {
        let started = Instant::now();
        let mut stats = FrameStats::default();
        self.ensure_targets(size);

        // --- 1. per-frame uniform ------------------------------------------
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let view = camera.view();
        let (near, far) = camera.near_far();
        let headlight = settings.headlight.unwrap_or(self.scene_headlight);
        // 🦀 Iterator chains: the headlight (if on) first, then scene lights.
        let lights: Vec<GpuLight> = headlight
            .then(|| Light::headlight(camera.forward()))
            .iter()
            .chain(&self.lights)
            .map(GpuLight::from)
            .collect();
        if lights.len() > MAX_LIGHTS && !std::mem::replace(&mut self.warned_lights, true) {
            log::warn!(
                "{} lights; the shader takes {MAX_LIGHTS}, the rest are ignored",
                lights.len()
            );
        }
        let mut frame = FrameUniform {
            view_proj: (camera.projection(aspect) * view).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            camera_pos: camera.eye().extend(settings.exposure).into(),
            params: [
                lights.len().min(MAX_LIGHTS) as u32,
                settings.view as u32,
                settings.color_mode as u32,
                u32::from(self.color_format.is_srgb()) | u32::from(headlight) << 1,
            ],
            near_far: [
                near,
                far,
                (camera.distance - camera.scene_radius).max(near),
                camera.distance + camera.scene_radius,
            ],
            viewport: [size.0 as f32, size.1 as f32, settings.line_scale, 0.0],
            lights: [bytemuck::Zeroable::zeroed(); MAX_LIGHTS],
        };
        for (slot, light) in frame.lights.iter_mut().zip(&lights) {
            *slot = *light;
        }
        stats.lights = frame.params[0];
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));

        // --- 2. order the draws --------------------------------------------
        let wireframe = settings.wireframe && self.supports_wireframe;
        let eye = camera.eye();
        let key_for = |d: &DrawItem, meshes: &[GpuMesh]| {
            let m = &meshes[d.mesh];
            PipelineKey {
                topology: m.topology(),
                blend: d.transparent,
                cull: settings.cull_back_faces && m.solid(),
                wireframe: wireframe && m.topology() == Topology::Triangles,
            }
        };
        let visible = |d: &&DrawItem| {
            let shown = match self.meshes[d.mesh].topology() {
                Topology::Triangles => true,
                Topology::Lines => settings.show_lines,
                Topology::Points => settings.show_points,
            };
            // X3D LOD: draw only the level the camera's distance picks.
            // …and only the chosen child of each animated Switch.
            let switched_on = self.guards[d.guard]
                .iter()
                .all(|g| self.switches[g.switch as usize] == g.child as i32);
            shown && switched_on && self.lods[d.lod].iter().all(|t| t.active(eye))
        };
        let (mut transparent, opaque): (Vec<DrawItem>, Vec<DrawItem>) = self
            .draws
            .iter()
            .filter(visible)
            .partition(|d| d.transparent);
        // Back to front: farthest first. `total_cmp` sorts floats without
        // panicking on NaN (f32 is only `PartialOrd`).
        transparent.sort_by(|a, b| {
            b.center
                .distance_squared(eye)
                .total_cmp(&a.center.distance_squared(eye))
        });
        let ordered: Vec<(DrawItem, PipelineKey)> = opaque
            .into_iter()
            .chain(transparent)
            .map(|d| (d, key_for(&d, &self.meshes)))
            .collect();

        // Build any pipelines this frame needs (before the pass borrows them).
        for (_, key) in &ordered {
            if !self.pipelines.contains_key(key) {
                let p = self.create_pipeline(&self.shader, *key);
                self.pipelines.insert(*key, p);
            }
        }

        // --- 3. record the render pass -------------------------------------
        let targets = self.targets.as_ref().expect("ensure_targets ran");
        let clear = clear_color(&self.background, self.color_format.is_srgb());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            // With MSAA we draw into the multisampled texture and *resolve*
            // (average) it into the real target at the end of the pass.
            let (view, resolve_target, store) = match &targets.msaa {
                Some(msaa) => (msaa, Some(target), wgpu::StoreOp::Discard),
                None => (target, None, wgpu::StoreOp::Store),
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            let mut current: Option<PipelineKey> = None;
            for (item, key) in &ordered {
                if current != Some(*key) {
                    pass.set_pipeline(&self.pipelines[key]);
                    current = Some(*key);
                    stats.pipeline_switches += 1;
                }
                pass.set_bind_group(1, &self.draw_bind_group, &[item.offset]);
                pass.set_bind_group(2, &self.materials[item.material].bind_group, &[]);
                match &self.meshes[item.mesh] {
                    GpuMesh::Triangles {
                        vertices,
                        indices,
                        index_count,
                        ..
                    } => {
                        pass.set_vertex_buffer(0, vertices.slice(..));
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                    // 6 vertices (one quad) × one instance per segment.
                    GpuMesh::Wide {
                        segments, count, ..
                    } => {
                        pass.set_vertex_buffer(0, segments.slice(..));
                        pass.draw(0..6, 0..*count);
                    }
                }
                stats.draw_calls += 1;
            }
        } // 🦀 the pass borrows the encoder; it must end (drop) before `finish`

        // --- 4. submit -----------------------------------------------------
        self.queue.submit([encoder.finish()]);
        stats.cpu_ms = started.elapsed().as_secs_f32() * 1000.0;
        stats
    }
}

/// Index of a LOD tag list in `lods` (0 = empty), adding it if new.
fn intern_lod(lods: &mut Vec<Arc<[LodTag]>>, tags: &Arc<[LodTag]>) -> usize {
    if tags.is_empty() {
        return 0;
    }
    lods.push(Arc::clone(tags));
    lods.len() - 1
}

/// Index of a Switch guard list in `guards` (0 = empty), adding it if new.
fn intern_guard(guards: &mut Vec<Arc<[SwitchGuard]>>, list: &Arc<[SwitchGuard]>) -> usize {
    if list.is_empty() {
        return 0;
    }
    guards.push(Arc::clone(list));
    guards.len() - 1
}

fn upload(device: &wgpu::Device, mesh: &Mesh) -> GpuMesh {
    let label = mesh.label.as_str();
    let buffer = |contents: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let v = |i: u32| mesh.vertices[i as usize];
    let segment = |a: u32, b: u32| Segment {
        a: v(a).position,
        b: v(b).position,
        color_a: v(a).color,
        color_b: v(b).color,
    };
    match mesh.topology {
        Topology::Triangles => GpuMesh::Triangles {
            vertices: buffer(
                bytemuck::cast_slice(&mesh.vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            indices: buffer(
                bytemuck::cast_slice(&mesh.indices),
                wgpu::BufferUsages::INDEX,
            ),
            index_count: mesh.indices.len() as u32,
            solid: mesh.solid,
        },
        topology => {
            let segments: Vec<Segment> = match topology {
                Topology::Lines => mesh
                    .indices
                    .chunks_exact(2)
                    .map(|p| segment(p[0], p[1]))
                    .collect(),
                _ => mesh.indices.iter().map(|&i| segment(i, i)).collect(),
            };
            GpuMesh::Wide {
                segments: buffer(bytemuck::cast_slice(&segments), wgpu::BufferUsages::VERTEX),
                count: segments.len() as u32,
                topology,
            }
        }
    }
}

/// The background colour as a clear value. Clear colours are written as-is
/// into a unorm target, but *encoded* by an sRGB one, so decode first there.
fn clear_color(bg: &Background, target_is_srgb: bool) -> wgpu::Color {
    let c = bg.clear_color();
    let decode = |x: f32| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    };
    let ch = |x: f32| f64::from(if target_is_srgb { decode(x) } else { x });
    wgpu::Color {
        r: ch(c.x),
        g: ch(c.y),
        b: ch(c.z),
        a: f64::from(c.w),
    }
}

/// Fall back to fewer MSAA samples if the GPU can't do what was asked.
fn supported_samples(gpu: &Gpu, format: wgpu::TextureFormat, wanted: u32) -> u32 {
    let ok = |f: wgpu::TextureFormat, n: u32| {
        gpu.adapter
            .get_texture_format_features(f)
            .flags
            .sample_count_supported(n)
    };
    match wanted {
        0 | 1 => 1,
        n if ok(format, n) && ok(DEPTH_FORMAT, n) => n,
        n => {
            log::warn!("{n}× MSAA not supported for {format:?}; using 1×");
            1
        }
    }
}
