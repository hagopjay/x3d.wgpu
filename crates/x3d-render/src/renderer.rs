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

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use x3d_scene::runtime::{Changes, Value};
use x3d_scene::{
    Aabb, Background, Draw, Light, LightKind, LodTag, Mesh, Scene, ShaderSource, SwitchGuard, Topology, Vertex,
};

use crate::camera::OrbitCamera;
use crate::custom::{self, Primitive, Program};
use crate::gpu::Gpu;
use crate::post::{HDR_FORMAT, Post};
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
    pub const ALL: [Self; 7] =
        [Self::Lit, Self::Normals, Self::Depth, Self::BaseColor, Self::LightingOnly, Self::Facing, Self::Uv];

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

/// The curve that squeezes HDR light into what a screen can show.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ToneMap {
    /// No curve: values above 1 clip (X_ITE's look).
    #[default]
    Off = 0,
    /// The filmic S-curve most games used for a decade (Narkowicz fit).
    Aces = 1,
    /// Blender's default since 4.0: bright colours drift to white, not to yellow.
    Agx = 2,
}

impl ToneMap {
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::Aces,
            Self::Aces => Self::Agx,
            Self::Agx => Self::Off,
        }
    }
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
    /// Tone mapping; anything but `Off` turns on the HDR path (see `post`),
    /// and then `exposure` scales *all* light, not only PhysicalMaterial's.
    pub tone_map: ToneMap,
    /// How much bright light bleeds into its surroundings (0 = no bloom).
    pub bloom: f32,
    /// Linear brightness where bloom starts (soft-kneed).
    pub bloom_threshold: f32,
    /// Sun shadows: the first DirectionalLight casts a shadow map.
    pub shadows: bool,
}

impl RenderSettings {
    /// True when the scene renders to a float target and goes through `post`.
    pub fn hdr(&self) -> bool {
        self.tone_map != ToneMap::Off || self.bloom > 0.0
    }
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
            tone_map: ToneMap::Off,
            bloom: 0.0,
            bloom_threshold: 0.8,
            shadows: false,
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
        /// The raw vertices and indices, for ComposedShaders (which draw
        /// points as sprites and lines as plain lines).
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        index_count: u32,
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
            Self::Triangles { vertices, indices, .. } => vertices.size() + indices.size(),
            Self::Wide { segments, vertices, indices, .. } => segments.size() + vertices.size() + indices.size(),
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
    /// World matrix (ComposedShaders need it every frame).
    model: glam::Mat4,
    /// A ComposedShader: (program, slot in its uniform buffer).
    custom: Option<(usize, u32)>,
}

/// Every distinct pipeline we might need.
/// 🦀 `#[derive(Hash, Eq)]` is all it takes to use a struct as a map key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineKey {
    topology: Topology,
    blend: bool,
    cull: bool,
    wireframe: bool,
    /// A ComposedShader program (index into `Renderer::programs`) instead
    /// of the built-in shader.
    program: Option<usize>,
    /// Drawing into the float HDR target instead of the output format.
    hdr: bool,
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
    shadow: ShadowMap,
    /// The scene's world bounds: what the sun's shadow map has to cover.
    bounds: Aabb,
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
    /// Compiled ComposedShaders, and which (shader, sprites) each one is.
    programs: Vec<Program>,
    program_of: HashMap<(u32, bool), Option<usize>>,
    shader_sources: Vec<ShaderSource>,
    shader_fields: Vec<Value>,
    lights: Vec<Light>,
    /// Bloom + tone mapping, created the first time HDR is switched on.
    post: Option<Post>,
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
            entries: &[
                uniform_entry(false, size_of::<FrameUniform>()),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw layout"),
            entries: &[uniform_entry(true, size_of::<DrawUniform>())],
        });
        let textures = TextureCache::new(&device, &gpu.queue);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(&draw_layout), Some(textures.layout())],
            immediate_size: 0,
        });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame uniform"),
            size: size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow = ShadowMap::new(&device, &frame_buffer, &draw_layout);
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame bind group"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: frame_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shadow.view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shadow.sampler) },
            ],
        });

        let align = device.limits().min_uniform_buffer_offset_alignment;
        let draw_stride = (size_of::<DrawUniform>() as u32).next_multiple_of(align);
        let (draw_buffer, draw_bind_group) = Self::draw_buffer(&device, &draw_layout, draw_stride, 1);

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
            shadow,
            bounds: Aabb::EMPTY,
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
            programs: Vec::new(),
            program_of: HashMap::new(),
            shader_sources: Vec::new(),
            shader_fields: Vec::new(),
            lights: Vec::new(),
            background: Background::default(),
            scene_headlight: true,
            targets: None,
            post: None,
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
        self.programs.clear();
        self.program_of.clear();
        self.shader_sources.clone_from(&scene.shaders);
        self.shader_fields.clone_from(&scene.shader_fields);
        self.bounds = scene.bounds();
        let mut slots: Vec<u32> = Vec::new();
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
            let material = *material_of.entry(Arc::as_ptr(&textures)).or_insert_with(|| {
                self.materials.push(textures);
                self.materials.len() - 1
            });
            // 🦀 `fold` builds the bit mask: bit i = scene light i reaches this draw.
            let light_mask = scene
                .lights
                .iter()
                .take(32)
                .enumerate()
                .fold(0u32, |mask, (i, l)| if l.reaches(&draw.scopes) { mask | 1 << i } else { mask });
            // A ComposedShader? Compile it once per (shader, sprites).
            let custom = draw.material.shader.and_then(|id| {
                let sprites = mesh.topology == Topology::Points;
                let program = (*self.program_of.entry((id, sprites)).or_insert_with(|| {
                    let src = &scene.shaders[id as usize];
                    let built =
                        custom::compile(src, sprites).and_then(|c| Program::new(&self.device, c, src.label.clone(), 1));
                    match built {
                        Ok(p) => {
                            self.programs.push(p);
                            slots.push(0);
                            Some(self.programs.len() - 1)
                        }
                        Err(e) => {
                            log::warn!("{} falls back to the built-in shading:\n{e}", src.label);
                            None
                        }
                    }
                }))?;
                slots[program] += 1;
                Some((program, slots[program] - 1))
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
                model: draw.transform,
                custom,
            });
        }
        if (self.draw_stride as usize * self.draws.len().max(1)) as u64 > self.draw_buffer.size() {
            (self.draw_buffer, self.draw_bind_group) =
                Self::draw_buffer(&self.device, &self.draw_layout, self.draw_stride, self.draws.len());
        }
        self.write_draw_uniforms(scene);
        for (program, n) in self.programs.iter_mut().zip(&slots) {
            program.reserve(&self.device, *n as usize);
        }

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
        if changes.shader_fields {
            self.shader_fields.clone_from(&scene.shader_fields);
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
            // X_ITE's rule: blended if the material is transparent, the base
            // texture has alpha, or the colours come from a ColorRGBA node.
            item.transparent = draw.material.is_transparent() || item.texture_alpha || draw.mesh.rgba_colors;
            item.center = draw.world_bounds().center();
            item.model = draw.transform;
        }
        self.queue.write_buffer(&self.draw_buffer, 0, &bytes);
    }

    /// Replace the shader. On any compile/validation error the old shader
    /// keeps running and the error text comes back as `Err`.
    pub fn reload_shader(&mut self, source: &str) -> Result<(), String> {
        // 🎨 Error scopes catch validation errors instead of sending them to
        // the global handler, so we can decide what to do with them.
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
            program: None,
            hdr: false,
        };
        let probe = self.create_pipeline(&module, key);
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(err.to_string());
        }
        self.shader = module;
        self.shadow.pipeline = None;
        self.pipelines.clear();
        self.pipelines.insert(key, probe);
        Ok(())
    }

    fn create_pipeline(&self, shader: &wgpu::ShaderModule, key: PipelineKey) -> wgpu::RenderPipeline {
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
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                polygon_mode: if key.wireframe { wgpu::PolygonMode::Line } else { wgpu::PolygonMode::Fill },
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
            multisample: wgpu::MultisampleState { count: self.sample_count, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: if key.hdr { HDR_FORMAT } else { self.color_format },
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
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
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
        let msaa = (self.sample_count > 1).then(|| texture("msaa colour", self.color_format, self.sample_count));
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
        let hdr = settings.hdr();
        if hdr {
            let post = self
                .post
                .get_or_insert_with(|| Post::new(&self.device, &self.queue, self.color_format, self.sample_count));
            post.ensure(&self.device, size);
        }
        // In HDR the scene writes display values unclipped and unencoded;
        // exposure is applied once, by the tone-mapping pass.
        let srgb_out = self.color_format.is_srgb() && !hdr;
        let exposure = if hdr { 1.0 } else { settings.exposure };

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
        // The sun: the first DirectionalLight that is on (and fits in the array).
        let sun = settings
            .shadows
            .then(|| {
                self.lights.iter().enumerate().find_map(|(i, l)| match l.kind {
                    LightKind::Directional { direction } if l.on => Some((i + usize::from(headlight), direction)),
                    _ => None,
                })
            })
            .flatten()
            .filter(|(i, _)| *i < MAX_LIGHTS);
        let shadow_proj = sun.map_or(Mat4::IDENTITY, |(_, d)| sun_projection(&self.bounds, d));
        let shadow_texel = self.bounds.size().length() / SHADOW_SIZE as f32;
        if lights.len() > MAX_LIGHTS && !std::mem::replace(&mut self.warned_lights, true) {
            log::warn!("{} lights; the shader takes {MAX_LIGHTS}, the rest are ignored", lights.len());
        }
        let mut frame = FrameUniform {
            view_proj: (camera.projection(aspect) * view).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            camera_pos: camera.eye().extend(exposure).into(),
            params: [
                lights.len().min(MAX_LIGHTS) as u32,
                settings.view as u32,
                settings.color_mode as u32,
                u32::from(srgb_out) | u32::from(headlight) << 1 | u32::from(hdr) << 2,
            ],
            near_far: [
                near,
                far,
                (camera.distance - camera.scene_radius).max(near),
                camera.distance + camera.scene_radius,
            ],
            viewport: [size.0 as f32, size.1 as f32, settings.line_scale, 0.0],
            shadow_proj: shadow_proj.to_cols_array_2d(),
            shadow: [sun.map_or(-1.0, |(i, _)| i as f32), 0.0005, 1.0 / SHADOW_SIZE as f32, shadow_texel],
            lights: [bytemuck::Zeroable::zeroed(); MAX_LIGHTS],
        };
        for (slot, light) in frame.lights.iter_mut().zip(&lights) {
            *slot = *light;
        }
        stats.lights = frame.params[0];
        self.queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));

        // --- 2. order the draws --------------------------------------------
        let wireframe = settings.wireframe && self.supports_wireframe;
        let eye = camera.eye();
        let key_for = |d: &DrawItem, meshes: &[GpuMesh]| {
            let m = &meshes[d.mesh];
            PipelineKey {
                topology: m.topology(),
                blend: d.transparent,
                cull: settings.cull_back_faces && m.solid(),
                wireframe: wireframe && m.topology() == Topology::Triangles && d.custom.is_none(),
                program: d.custom.map(|(p, _)| p),
                hdr,
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
            let switched_on = self.guards[d.guard].iter().all(|g| self.switches[g.switch as usize] == g.child as i32);
            shown && switched_on && self.lods[d.lod].iter().all(|t| t.active(eye))
        };
        let (mut transparent, opaque): (Vec<DrawItem>, Vec<DrawItem>) =
            self.draws.iter().filter(visible).partition(|d| d.transparent);
        // Back to front: farthest first. `total_cmp` sorts floats without
        // panicking on NaN (f32 is only `PartialOrd`).
        transparent.sort_by(|a, b| b.center.distance_squared(eye).total_cmp(&a.center.distance_squared(eye)));
        let ordered: Vec<(DrawItem, PipelineKey)> =
            opaque.into_iter().chain(transparent).map(|d| (d, key_for(&d, &self.meshes))).collect();

        // Build any pipelines this frame needs (before the pass borrows them).
        let mut broken: Vec<PipelineKey> = Vec::new();
        for (_, key) in &ordered {
            match key.program {
                Some(p) => {
                    let format = if hdr { HDR_FORMAT } else { self.color_format };
                    let samples = self.sample_count;
                    let device = &self.device;
                    if self.programs[p]
                        .pipeline(device, primitive(key), key.blend, format, DEPTH_FORMAT, samples)
                        .is_none()
                    {
                        broken.push(*key);
                    }
                }
                None if !self.pipelines.contains_key(key) => {
                    let p = self.create_pipeline(&self.shader, *key);
                    self.pipelines.insert(*key, p);
                }
                None => {}
            }
        }
        let ordered: Vec<(DrawItem, PipelineKey)> = ordered.into_iter().filter(|(_, k)| !broken.contains(k)).collect();

        // ComposedShader uniforms: X_ITE's built-ins and the fields, per draw.
        let projection = camera.projection(aspect);
        let mut field_cache: HashMap<usize, Vec<(String, Value)>> = HashMap::new();
        for (item, _) in &ordered {
            let Some((p, slot)) = item.custom else { continue };
            let fields = field_cache.entry(p).or_insert_with(|| {
                let id = self.program_of.iter().find(|(_, v)| **v == Some(p)).map_or(0, |((id, _), _)| *id);
                custom::field_values(&self.shader_sources[id as usize], &self.shader_fields)
            });
            let inputs = custom::Inputs { projection, view, model: item.model, viewport: size, fields };
            let program = &mut self.programs[p];
            let (stride, len) = (program.stride as usize, program.block.size as usize);
            let at = slot as usize * stride;
            let block = program.block.clone();
            block.fill(&mut program.staging[at..at + len], &inputs);
        }
        for program in &self.programs {
            self.queue.write_buffer(&program.buffer, 0, &program.staging);
        }

        // --- 3. record the render pass -------------------------------------
        let targets = self.targets.as_ref().expect("ensure_targets ran");
        let clear = clear_color(&self.background, srgb_out);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        if sun.is_some() {
            if self.shadow.pipeline.is_none() {
                self.shadow.pipeline = Some(self.shadow.pipeline(&self.device, &self.shader));
            }
            let casters = ordered.iter().filter(|(_, k)| k.topology == Topology::Triangles && !k.blend);
            let meshes = casters.map(|(item, _)| (item.offset, &self.meshes[item.mesh]));
            self.shadow.record(&mut encoder, &self.draw_bind_group, meshes);
        }
        {
            // With MSAA we draw into the multisampled texture and *resolve*
            // (average) it into the real target at the end of the pass.
            let (view, resolve_target, store) = match (&self.post, &targets.msaa) {
                (Some(post), _) if hdr => match post.scene_targets() {
                    (msaa, Some(resolve)) => (msaa, Some(resolve), wgpu::StoreOp::Discard),
                    (view, None) => (view, None, wgpu::StoreOp::Store),
                },
                (_, Some(msaa)) => (msaa, Some(target), wgpu::StoreOp::Discard),
                (_, None) => (target, None, wgpu::StoreOp::Store),
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            let mut current: Option<PipelineKey> = None;
            for (item, key) in &ordered {
                if let Some((p, slot)) = item.custom {
                    // A ComposedShader: its own pipeline layout and uniform block.
                    let program = &self.programs[p];
                    if current != Some(*key) {
                        let format = if key.hdr { HDR_FORMAT } else { self.color_format };
                        let Some(pipeline) = program.get(primitive(key), key.blend, format) else { continue };
                        pass.set_pipeline(pipeline);
                        current = Some(*key);
                        stats.pipeline_switches += 1;
                    }
                    pass.set_bind_group(0, &program.bind_group, &[slot * program.stride]);
                    match &self.meshes[item.mesh] {
                        GpuMesh::Triangles { vertices, indices, index_count, .. }
                        | GpuMesh::Wide { vertices, indices, index_count, topology: Topology::Lines, .. } => {
                            pass.set_vertex_buffer(0, vertices.slice(..));
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*index_count, 0, 0..1);
                        }
                        // Sprites: 6 vertices per point, the point as the instance.
                        GpuMesh::Wide { vertices, index_count, .. } => {
                            pass.set_vertex_buffer(0, vertices.slice(..));
                            pass.draw(0..6, 0..*index_count);
                        }
                    }
                    stats.draw_calls += 1;
                    continue;
                }
                if current != Some(*key) {
                    pass.set_pipeline(&self.pipelines[key]);
                    // A ComposedShader may have used group 0: rebind the frame.
                    pass.set_bind_group(0, &self.frame_bind_group, &[]);
                    current = Some(*key);
                    stats.pipeline_switches += 1;
                }
                pass.set_bind_group(1, &self.draw_bind_group, &[item.offset]);
                pass.set_bind_group(2, &self.materials[item.material].bind_group, &[]);
                match &self.meshes[item.mesh] {
                    GpuMesh::Triangles { vertices, indices, index_count, .. } => {
                        pass.set_vertex_buffer(0, vertices.slice(..));
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*index_count, 0, 0..1);
                    }
                    // 6 vertices (one quad) × one instance per segment.
                    GpuMesh::Wide { segments, count, .. } => {
                        pass.set_vertex_buffer(0, segments.slice(..));
                        pass.draw(0..6, 0..*count);
                    }
                }
                stats.draw_calls += 1;
            }
        } // 🦀 the pass borrows the encoder; it must end (drop) before `finish`

        if let Some(post) = self.post.as_ref().filter(|_| hdr) {
            post.run(&self.queue, &mut encoder, target, settings);
        }

        // --- 4. submit -----------------------------------------------------
        self.queue.submit([encoder.finish()]);
        stats.cpu_ms = started.elapsed().as_secs_f32() * 1000.0;
        stats
    }
}

/// How a ComposedShader draws this key's geometry.
fn primitive(key: &PipelineKey) -> Primitive {
    match key.topology {
        Topology::Triangles => Primitive::Triangles { cull: key.cull },
        Topology::Lines => Primitive::Lines,
        Topology::Points => Primitive::Sprites,
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
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
    };
    let v = |i: u32| mesh.vertices[i as usize];
    let segment =
        |a: u32, b: u32| Segment { a: v(a).position, b: v(b).position, color_a: v(a).color, color_b: v(b).color };
    match mesh.topology {
        Topology::Triangles => GpuMesh::Triangles {
            vertices: buffer(bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
            indices: buffer(bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
            index_count: mesh.indices.len() as u32,
            solid: mesh.solid,
        },
        topology => {
            let segments: Vec<Segment> = match topology {
                Topology::Lines => mesh.indices.chunks_exact(2).map(|p| segment(p[0], p[1])).collect(),
                _ => mesh.indices.iter().map(|&i| segment(i, i)).collect(),
            };
            GpuMesh::Wide {
                segments: buffer(bytemuck::cast_slice(&segments), wgpu::BufferUsages::VERTEX),
                count: segments.len() as u32,
                topology,
                vertices: buffer(bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
                indices: buffer(bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
                index_count: mesh.indices.len() as u32,
            }
        }
    }
}

/// The background colour as a clear value. Clear colours are written as-is
/// into a unorm target, but *encoded* by an sRGB one, so decode first there.
fn clear_color(bg: &Background, target_is_srgb: bool) -> wgpu::Color {
    let c = bg.clear_color();
    let decode = |x: f32| {
        if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
    };
    let ch = |x: f32| f64::from(if target_is_srgb { decode(x) } else { x });
    wgpu::Color { r: ch(c.x), g: ch(c.y), b: ch(c.z), a: f64::from(c.w) }
}

/// Fall back to fewer MSAA samples if the GPU can't do what was asked.
fn supported_samples(gpu: &Gpu, format: wgpu::TextureFormat, wanted: u32) -> u32 {
    let ok =
        |f: wgpu::TextureFormat, n: u32| gpu.adapter.get_texture_format_features(f).flags.sample_count_supported(n);
    match wanted {
        0 | 1 => 1,
        n if ok(format, n) && ok(DEPTH_FORMAT, n) => n,
        n => {
            log::warn!("{n}× MSAA not supported for {format:?}; using 1×");
            1
        }
    }
}

// --- sun shadows -------------------------------------------------------------

/// Shadow-map resolution (square). 2048² Depth32 = 16 MiB.
pub const SHADOW_SIZE: u32 = 2048;

/// The sun's view of the scene: an orthographic box (sunlight is parallel)
/// around the scene's bounding sphere, looking along the light.
fn sun_projection(bounds: &Aabb, direction: Vec3) -> Mat4 {
    let center = bounds.center();
    let r = (bounds.size().length() * 0.5).max(1e-3);
    let d = direction.normalize_or(Vec3::NEG_Y);
    let up = if d.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let view = glam::camera::rh::view::look_at_mat4(center - d * 2.0 * r, center, up);
    glam::camera::rh::proj::directx::orthographic(-r, r, -r, r, 0.5 * r, 3.5 * r) * view
}

/// 🎨 The depth texture, its comparison sampler, and the depth-only
/// pipeline that fills it.
#[derive(Debug)]
struct ShadowMap {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Group 0 for the shadow pass: the frame uniform *without* the map,
    /// since a texture can't be read and written in the same pass.
    frame_bind_group: wgpu::BindGroup,
    layout: wgpu::PipelineLayout,
    pipeline: Option<wgpu::RenderPipeline>,
}

impl ShadowMap {
    fn new(device: &wgpu::Device, frame_buffer: &wgpu::Buffer, draw_layout: &wgpu::BindGroupLayout) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow map"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow frame layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self {
            view: texture.create_view(&Default::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }),
            frame_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow frame"),
                layout: &frame_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: frame_buffer.as_entire_binding() }],
            }),
            layout: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[Some(&frame_layout), Some(draw_layout)],
                immediate_size: 0,
            }),
            pipeline: None,
        }
    }

    fn pipeline(&self, device: &wgpu::Device, shader: &wgpu::ShaderModule) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                // Same vertex buffers as the scene, but only the position is read.
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            // No culling: thin single-sided X3D geometry should still cast.
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                // Slope-scaled bias: surfaces seen edge-on by the sun get more.
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        })
    }

    /// Draw every caster's depth from the sun into the map.
    fn record<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        draws: &wgpu::BindGroup,
        casters: impl Iterator<Item = (u32, &'a GpuMesh)>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(self.pipeline.as_ref().expect("shadow pipeline built"));
        pass.set_bind_group(0, &self.frame_bind_group, &[]);
        for (offset, mesh) in casters {
            if let GpuMesh::Triangles { vertices, indices, index_count, .. } = mesh {
                pass.set_bind_group(1, draws, &[offset]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
        }
    }
}
