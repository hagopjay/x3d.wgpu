//! Bloom and tone mapping: the passes that run *after* the scene.
//!
//! 🎨 **Why a float target?** An 8-bit target clips every value at 1.0, so
//! a white-hot sprite and a merely white wall look the same and nothing is
//! left to glow. With post-processing on, the scene draws into an
//! `Rgba16Float` texture instead; [`Post::run`] then blurs the bright parts
//! (bloom), scales by exposure, and squeezes the result into 0..1 with a
//! tone-mapping curve. The shaders are in `shaders/post.wgsl`.
//!
//! Bloom works on a *mip chain*: halve the image several times (each step
//! blurs a little), then walk back up adding each level onto the next
//! larger one. Small mips spread light far; large mips keep it tight.

use crate::renderer::RenderSettings;
use crate::timing::{self, GpuTimer};

/// The format the scene renders into when post-processing is on.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The most bloom mips (mip 0 is half the screen; each next one half again).
const MAX_MIPS: u32 = 6;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostUniform {
    /// strength, threshold, knee, exposure
    bloom: [f32; 4],
    /// tone map, output is sRGB, -, -
    mode: [u32; 4],
}

#[derive(Debug)]
struct Sized {
    size: (u32, u32),
    /// The scene's colour (resolved, single-sample).
    hdr: wgpu::TextureView,
    /// The multisampled scene colour, when MSAA is on.
    hdr_msaa: Option<wgpu::TextureView>,
    /// One view per bloom mip.
    mips: Vec<wgpu::TextureView>,
    prefilter: wgpu::BindGroup,
    /// `down[i]` reads mip i and writes mip i+1.
    down: Vec<wgpu::BindGroup>,
    /// `up[i]` reads mip i+1 and adds into mip i.
    up: Vec<wgpu::BindGroup>,
    /// Reads the scene and mip 0 (or a black texel if bloom is off).
    composite: wgpu::BindGroup,
    composite_no_bloom: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct Post {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    black: wgpu::TextureView,
    prefilter: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    output_format: wgpu::TextureFormat,
    samples: u32,
    sized: Option<Sized>,
}

impl Post {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_format: wgpu::TextureFormat,
        samples: u32,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/post.wgsl"));
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(3),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, format, blend| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache,
            })
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let black = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("black"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            black.as_image_copy(),
            &[0u8; 8],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: None },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        Self {
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("post"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("post"),
                size: size_of::<PostUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            black: black.create_view(&Default::default()),
            prefilter: pipeline("fs_prefilter", HDR_FORMAT, None),
            down: pipeline("fs_down", HDR_FORMAT, None),
            up: pipeline("fs_up", HDR_FORMAT, Some(additive)),
            composite: pipeline("fs_composite", output_format, None),
            layout,
            output_format,
            samples,
            sized: None,
        }
    }

    /// The view the scene pass draws into, and the one it resolves into.
    pub fn scene_targets(&self) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        let s = self.sized.as_ref().expect("Post::ensure ran");
        match &s.hdr_msaa {
            Some(msaa) => (msaa, Some(&s.hdr)),
            None => (&s.hdr, None),
        }
    }

    /// (Re)create the textures for a `size` frame.
    pub fn ensure(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if self.sized.as_ref().is_some_and(|s| s.size == size) {
            return;
        }
        let texture = |label, (w, h): (u32, u32), mips, samples| {
            let usage = if samples > 1 {
                wgpu::TextureUsages::RENDER_ATTACHMENT
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            };
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: mips,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let hdr = texture("hdr scene", size, 1, 1).create_view(&Default::default());
        let hdr_msaa = (self.samples > 1)
            .then(|| texture("hdr scene msaa", size, 1, self.samples).create_view(&Default::default()));

        // Stop halving while the smallest side is still ≥ 2 pixels.
        let half = (size.0 / 2, size.1 / 2);
        let levels = (half.0.min(half.1).max(1).ilog2()).clamp(1, MAX_MIPS);
        let chain = texture("bloom", half, levels, 1);
        let mips: Vec<_> = (0..levels)
            .map(|m| {
                chain.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let group = |source: &wgpu::TextureView, extra: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("post"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(extra) },
                ],
            })
        };
        let pairs = mips.windows(2);
        self.sized = Some(Sized {
            size,
            prefilter: group(&hdr, &self.black),
            down: pairs.clone().map(|w| group(&w[0], &self.black)).collect(),
            up: pairs.map(|w| group(&w[1], &self.black)).collect(),
            composite: group(&hdr, &mips[0]),
            composite_no_bloom: group(&hdr, &self.black),
            hdr,
            hdr_msaa,
            mips,
        });
    }

    /// Record bloom + tone mapping, reading the scene and writing `target`.
    pub fn run(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        settings: &RenderSettings,
        mut timer: Option<&mut GpuTimer>,
    ) {
        let s = self.sized.as_ref().expect("Post::ensure ran");
        let bloom = settings.bloom > 0.0;
        let uniform = PostUniform {
            bloom: [settings.bloom, settings.bloom_threshold, 0.5, settings.exposure],
            mode: [settings.tone_map as u32, u32::from(self.output_format.is_srgb()), 0, 0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        let mut pass = |label,
                        view: &wgpu::TextureView,
                        pipeline,
                        group,
                        load,
                        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        };
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        if bloom {
            let (first, last) = timer.as_deref_mut().map_or((None, None), |t| t.span(timing::BLOOM));
            pass("bloom prefilter", &s.mips[0], &self.prefilter, &s.prefilter, clear, first);
            for (i, group) in s.down.iter().enumerate() {
                pass("bloom down", &s.mips[i + 1], &self.down, group, clear, None);
            }
            let n = s.up.len();
            for (i, group) in s.up.iter().enumerate().rev() {
                // The last up-pass (i == 0) closes the bloom span.
                let tw = if i == 0 { last.clone() } else { None };
                pass("bloom up", &s.mips[i], &self.up, group, wgpu::LoadOp::Load, tw);
            }
            if n == 0 {
                // A one-mip chain has no up-passes: close the span on an empty pass.
                pass("bloom end", &s.mips[0], &self.up, &s.prefilter, wgpu::LoadOp::Load, last);
            }
        }
        let group = if bloom { &s.composite } else { &s.composite_no_bloom };
        let tw = timer.and_then(|t| t.pass(timing::TONE_MAP));
        pass("tone map", target, &self.composite, group, clear, tw);
    }
}
