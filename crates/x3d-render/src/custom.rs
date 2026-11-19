//! Running X3D ComposedShaders: compile, reflect, fill uniforms, draw.
//!
//! 🎨 **The path of a shader.** `glsl::translate` rewrites the X3D GLSL
//! into Vulkan-style GLSL; naga's GLSL front end parses it into naga IR
//! (`naga::Module`); naga's validator checks it; wgpu takes the IR as is
//! (`ShaderSource::Naga`) and turns it into SPIR-V for Vulkan (or MSL, HLSL…).
//!
//! 🎨 **Reflection.** We never hard-code where `time` lives in the uniform
//! block: naga already computed every member's byte offset, so we read them
//! from the module and write each value at its offset, by name.

use std::borrow::Cow;
use std::collections::HashMap;

use glam::{Mat3, Mat4};
use x3d_scene::ShaderSource;
use x3d_scene::runtime::Value;

use crate::glsl;

/// What a uniform block member holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Float,
    Int,
    Uint,
    /// floatN, N = 2..4
    Vec(u8),
    /// intN, N = 2..4
    IVec(u8),
    /// matN×N (columns × rows), each column 16-byte aligned
    Mat(u8, u8),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Member {
    pub name: String,
    pub offset: u32,
    pub kind: Kind,
}

/// The uniform block's layout, read from the compiled module.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub size: u32,
    pub members: Vec<Member>,
}

/// A translated, parsed and validated ComposedShader.
#[derive(Debug)]
pub struct Compiled {
    pub vertex: naga::Module,
    pub fragment: naga::Module,
    pub block: Block,
    pub translated: glsl::Translated,
}

/// Where a program's pipelines draw: colour and depth formats, MSAA count,
/// and the pipeline cache (if any).
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub format: wgpu::TextureFormat,
    pub depth: wgpu::TextureFormat,
    pub samples: u32,
    pub cache: Option<&'a wgpu::PipelineCache>,
}

/// Translate + parse + validate + reflect. `sprites`: drawn on a PointSet.
/// The error text is ready to print (naga's errors point at the line).
pub fn compile(src: &ShaderSource, sprites: bool) -> Result<Compiled, String> {
    let t = glsl::translate(&src.vertex, &src.fragment, sprites).map_err(|e| e.to_string())?;
    let vertex = parse(naga::ShaderStage::Vertex, &t.vertex).map_err(|e| format!("vertex shader:\n{e}"))?;
    let fragment = parse(naga::ShaderStage::Fragment, &t.fragment).map_err(|e| format!("fragment shader:\n{e}"))?;
    let block = reflect(&vertex)?;
    Ok(Compiled { vertex, fragment, block, translated: t })
}

fn parse(stage: naga::ShaderStage, code: &str) -> Result<naga::Module, String> {
    let module = naga::front::glsl::Frontend::default()
        .parse(&naga::front::glsl::Options::from(stage), code)
        .map_err(|e| e.emit_to_string(code))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| e.emit_to_string(code))?;
    Ok(module)
}

/// Find the `X3DUniforms` block (set 0, binding 0) and list its members.
fn reflect(module: &naga::Module) -> Result<Block, String> {
    use naga::{ScalarKind, TypeInner, VectorSize};
    let block = module
        .global_variables
        .iter()
        .map(|(_, v)| v)
        .find(|v| v.space == naga::AddressSpace::Uniform && v.binding.as_ref().is_some_and(|b| b.binding == 0));
    let Some(block) = block else {
        return Ok(Block { size: 16, members: Vec::new() }); // a shader with no uniforms at all
    };
    let TypeInner::Struct { members, span } = &module.types[block.ty].inner else {
        return Err("the uniform block isn't a struct".into());
    };
    let size = |s: VectorSize| s as u8;
    let members = members
        .iter()
        .map(|m| {
            let kind = match module.types[m.ty].inner {
                TypeInner::Scalar(s) => match s.kind {
                    ScalarKind::Float => Kind::Float,
                    ScalarKind::Sint => Kind::Int,
                    _ => Kind::Uint,
                },
                TypeInner::Vector { size: n, scalar } => match scalar.kind {
                    ScalarKind::Float => Kind::Vec(size(n)),
                    _ => Kind::IVec(size(n)),
                },
                TypeInner::Matrix { columns, rows, .. } => Kind::Mat(size(columns), size(rows)),
                ref other => return Err(format!("uniform of type {other:?} isn't supported")),
            };
            Ok(Member { name: m.name.clone().unwrap_or_default(), offset: m.offset, kind })
        })
        .collect::<Result<_, String>>()?;
    Ok(Block { size: (*span).max(16), members })
}

/// Everything a draw's block can be filled from.
#[derive(Debug)]
pub struct Inputs<'a> {
    pub projection: Mat4,
    pub view: Mat4,
    pub model: Mat4,
    pub viewport: (u32, u32),
    /// This shader's `<field>`s, by name, with their current values.
    pub fields: &'a [(String, Value)],
}

impl Block {
    /// Write one draw's uniforms into `out` (`self.size` bytes).
    pub fn fill(&self, out: &mut [u8], inputs: &Inputs<'_>) {
        out.fill(0);
        let model_view = inputs.view * inputs.model;
        for m in &self.members {
            let value = match m.name.as_str() {
                // X_ITE's built-in uniforms.
                "x3d_ProjectionMatrix" => Value4::M4(inputs.projection),
                "x3d_ModelViewMatrix" => Value4::M4(model_view),
                "x3d_ViewMatrix" => Value4::M4(inputs.view),
                "x3d_ModelMatrix" => Value4::M4(inputs.model),
                "x3d_CameraSpaceMatrix" => Value4::M4(inputs.view.inverse()),
                "x3d_NormalMatrix" => Value4::M3(Mat3::from_mat4(model_view).inverse().transpose()),
                "x3d_Viewport" => Value4::I([0, 0, inputs.viewport.0 as i32, inputs.viewport.1 as i32]),
                // The shader's own fields.
                name => match inputs.fields.iter().find(|(n, _)| n == name) {
                    Some((_, v)) => Value4::from(*v),
                    None => continue, // stays zero
                },
            };
            write(&mut out[m.offset as usize..], m.kind, value);
        }
    }
}

/// A value on its way into the block.
#[derive(Debug, Clone, Copy)]
enum Value4 {
    F([f32; 4]),
    I([i32; 4]),
    M3(Mat3),
    M4(Mat4),
}

impl From<Value> for Value4 {
    fn from(v: Value) -> Self {
        match v {
            Value::Float(f) => Self::F([f, 0.0, 0.0, 0.0]),
            // SFTime arrives in seconds; GLSL has only 32-bit floats.
            Value::Time(t) => Self::F([t as f32, 0.0, 0.0, 0.0]),
            Value::Int(i) => Self::I([i, 0, 0, 0]),
            Value::Bool(b) => Self::I([i32::from(b), 0, 0, 0]),
            Value::Vec2(v) => Self::F([v.x, v.y, 0.0, 0.0]),
            Value::Vec3(v) => Self::F(v.extend(0.0).into()),
            Value::Vec4(v) => Self::F(v.into()),
            // SFRotation → vec4(axis, angle), as X_ITE passes it.
            Value::Rotation(q) => {
                let (axis, angle) = q.to_axis_angle();
                Self::F(axis.extend(angle).into())
            }
        }
    }
}

fn write(out: &mut [u8], kind: Kind, value: Value4) {
    let put_f = |out: &mut [u8], at: usize, x: f32| out[at..at + 4].copy_from_slice(&x.to_ne_bytes());
    let put_i = |out: &mut [u8], at: usize, x: i32| out[at..at + 4].copy_from_slice(&x.to_ne_bytes());
    match (kind, value) {
        (Kind::Float, Value4::F(f)) => put_f(out, 0, f[0]),
        (Kind::Float, Value4::I(i)) => put_f(out, 0, i[0] as f32),
        (Kind::Int | Kind::Uint, Value4::I(i)) => put_i(out, 0, i[0]),
        (Kind::Int | Kind::Uint, Value4::F(f)) => put_i(out, 0, f[0] as i32),
        (Kind::Vec(n), Value4::F(f)) => (0..n as usize).for_each(|k| put_f(out, 4 * k, f[k])),
        (Kind::IVec(n), Value4::I(i)) => (0..n as usize).for_each(|k| put_i(out, 4 * k, i[k])),
        (Kind::Mat(c, r), Value4::M4(m)) => write_matrix(out, c, r, |col, row| m.col(col)[row]),
        (Kind::Mat(c, r), Value4::M3(m)) => write_matrix(out, c, r, |col, row| m.col(col)[row]),
        _ => {} // type mismatch: leave it zero
    }
}

/// Matrix columns are 16-byte aligned in uniform blocks (a mat3 is 3 × vec4).
fn write_matrix(out: &mut [u8], cols: u8, rows: u8, at: impl Fn(usize, usize) -> f32) {
    for c in 0..cols as usize {
        for r in 0..rows as usize {
            let i = 16 * c + 4 * r;
            out[i..i + 4].copy_from_slice(&at(c, r).to_ne_bytes());
        }
    }
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------

/// Which pipeline of a program: what's drawn and how it blends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Primitive {
    Triangles {
        cull: bool,
    },
    Lines,
    /// One quad per point (an instance), see the sprite wrapper in `glsl`.
    Sprites,
}

/// One compiled ComposedShader, ready to draw with.
#[derive(Debug)]
pub struct Program {
    pub label: String,
    vertex: wgpu::ShaderModule,
    fragment: wgpu::ShaderModule,
    pub block: Block,
    layout: wgpu::PipelineLayout,
    bind_layout: wgpu::BindGroupLayout,
    pipelines: HashMap<(Primitive, bool, wgpu::TextureFormat), wgpu::RenderPipeline>,
    /// One block per draw, `stride` bytes apart.
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub stride: u32,
    pub staging: Vec<u8>,
}

impl Program {
    /// Upload a compiled shader. Validation errors come back as text.
    pub fn new(device: &wgpu::Device, compiled: Compiled, label: String, slots: usize) -> Result<Self, String> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = |m: naga::Module, what: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{label} ({what})")),
                source: wgpu::ShaderSource::Naga(Cow::Owned(m)),
            })
        };
        let vertex = module(compiled.vertex, "vertex");
        let fragment = module(compiled.fragment, "fragment");
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&label),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(u64::from(compiled.block.size)),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&label),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let stride = compiled.block.size.next_multiple_of(device.limits().min_uniform_buffer_offset_alignment);
        let (buffer, bind_group) = uniform_buffer(device, &bind_layout, stride, compiled.block.size, slots);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        Ok(Self {
            label,
            vertex,
            fragment,
            block: compiled.block,
            layout,
            bind_layout,
            pipelines: HashMap::new(),
            buffer,
            bind_group,
            stride,
            staging: vec![0; stride as usize * slots.max(1)],
        })
    }

    /// Make room for `slots` draws.
    pub fn reserve(&mut self, device: &wgpu::Device, slots: usize) {
        let bytes = self.stride as usize * slots.max(1);
        if bytes as u64 > self.buffer.size() {
            (self.buffer, self.bind_group) =
                uniform_buffer(device, &self.bind_layout, self.stride, self.block.size, slots);
        }
        self.staging.resize(bytes, 0);
    }

    /// The pipeline for one way of drawing, built on first use. A pipeline
    /// that fails validation (say, an attribute type the shader can't take)
    /// returns `None` and the draw is skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline(
        &mut self,
        device: &wgpu::Device,
        primitive: Primitive,
        blend: bool,
        target: &Target<'_>,
    ) -> Option<&wgpu::RenderPipeline> {
        let format = target.format;
        if !self.pipelines.contains_key(&(primitive, blend, format)) {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = self.create(device, primitive, blend, target);
            if let Some(e) = pollster::block_on(scope.pop()) {
                log::warn!("{}: {e}", self.label);
                return None;
            }
            self.pipelines.insert((primitive, blend, format), pipeline);
        }
        self.pipelines.get(&(primitive, blend, format))
    }

    /// A pipeline already built by [`Program::pipeline`].
    pub fn get(&self, primitive: Primitive, blend: bool, format: wgpu::TextureFormat) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(&(primitive, blend, format))
    }

    fn create(
        &self,
        device: &wgpu::Device,
        primitive: Primitive,
        blend: bool,
        target: &Target<'_>,
    ) -> wgpu::RenderPipeline {
        let Target { format, depth, samples, cache } = *target;
        let (topology, step_mode, cull) = match primitive {
            Primitive::Triangles { cull } => {
                (wgpu::PrimitiveTopology::TriangleList, wgpu::VertexStepMode::Vertex, cull)
            }
            Primitive::Lines => (wgpu::PrimitiveTopology::LineList, wgpu::VertexStepMode::Vertex, false),
            Primitive::Sprites => (wgpu::PrimitiveTopology::TriangleList, wgpu::VertexStepMode::Instance, false),
        };
        // The scene's Vertex: position, normal, colour, texcoord. The shader's
        // `vec4 x3d_Vertex` reads a 3-float position (w becomes 1).
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<x3d_scene::Vertex>() as u64,
            step_mode,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x3
            ],
        };
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} {primitive:?}", self.label)),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: &self.vertex,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            primitive: wgpu::PrimitiveState {
                topology,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: cull.then_some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(!blend),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &self.fragment,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                    // Like X_ITE: an opaque shape's alpha output is ignored.
                    write_mask: if blend { wgpu::ColorWrites::ALL } else { wgpu::ColorWrites::COLOR },
                })],
            }),
            multiview_mask: None,
            cache,
        })
    }
}

fn uniform_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u32,
    size: u32,
    slots: usize,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ComposedShader uniforms"),
        size: u64::from(stride) * slots.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ComposedShader uniforms"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: wgpu::BufferSize::new(u64::from(size)),
            }),
        }],
    });
    (buffer, bind_group)
}

/// Field values for one shader, looked up from the scene.
pub fn field_values(shader: &ShaderSource, values: &[Value]) -> Vec<(String, Value)> {
    shader.fields.iter().map(|(n, i)| (n.clone(), values[*i as usize])).collect()
}
