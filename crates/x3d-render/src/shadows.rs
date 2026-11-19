//! Shadows: cascaded maps for the sun, perspective maps for spot lights,
//! six maps per point light.
//!
//! 🎨 **One idea, three shapes.** A shadow map is a depth picture taken
//! from the light. Sunlight is parallel, so its picture is *orthographic*;
//! a spot light is a camera with the cone as its field of view; a point
//! light sees all around, so it takes six 90° pictures (a cube).
//!
//! 🎨 **Cascades.** One sun map over a whole Giza-sized scene gives texels
//! metres wide near the camera. So the view is cut into up to four slices
//! (near ones thin, far ones deep: the "practical" split, a blend of
//! uniform and logarithmic), and each slice gets its own map.
//!
//! 🎨 **Shimmer.** If a cascade's box follows the camera smoothly, the
//! texel grid slides over the world and shadow edges crawl. Two fixes keep
//! it still: the box's *size* comes from a bounding sphere (it doesn't
//! change as the camera turns), rounded up; and its *position* moves in
//! whole texels, in a light space that never rotates with the camera.
//! `stable_cascade` does both; a test checks the snapping.

use glam::{Mat4, Vec3, Vec4Swizzles};
use x3d_scene::{Aabb, Light, LightKind};

use crate::budget::Budget;
use crate::camera::OrbitCamera;
use crate::uniforms::{MAX_SHADOW_MATS, SUN_CASCADES_MAX};

pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// The uniform holding "which shadow matrix this pass renders" is an array
/// of these, one 256-byte slot per matrix (the dynamic-offset alignment).
const PASS_STRIDE: u64 = 256;
/// How much of the practical split is logarithmic (the rest uniform).
const SPLIT_LAMBDA: f32 = 0.7;

/// The shadows of one frame: which lights, which matrices, which passes.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowPlan {
    pub mats: [Mat4; MAX_SHADOW_MATS],
    /// (index in the GPU light array, shadowIntensity)
    pub sun: Option<(usize, f32)>,
    pub cascades: u32,
    pub cascade_far: [f32; 4],
    pub cascade_texel: [f32; 4],
    /// (index in the GPU light array, first matrix) for each spot/point caster.
    pub local: Vec<(usize, usize)>,
    /// Matrix indices to render a depth pass for.
    pub passes: Vec<usize>,
    /// Lights that asked for shadows but found no layers left.
    pub dropped: Vec<usize>,
}

impl ShadowPlan {
    pub fn none() -> Self {
        Self {
            mats: [Mat4::IDENTITY; MAX_SHADOW_MATS],
            sun: None,
            cascades: 0,
            cascade_far: [0.0; 4],
            cascade_texel: [0.0; 4],
            local: Vec::new(),
            passes: Vec::new(),
            dropped: Vec::new(),
        }
    }
}

/// Which lights cast, and the matrices for each.
///
/// `lights` is the GPU light order (headlight first if on). Lights with X3D
/// `shadows='true'` cast; if none asks, the first DirectionalLight that is on
/// does (what `--shadows` meant in 0.3).
pub fn plan(lights: &[Light], camera: &OrbitCamera, aspect: f32, bounds: &Aabb, budget: &Budget) -> ShadowPlan {
    let mut plan = ShadowPlan::none();
    let asked: Vec<usize> = lights.iter().enumerate().filter(|(_, l)| l.on && l.shadows).map(|(i, _)| i).collect();
    let casters = if asked.is_empty() {
        lights.iter().position(|l| l.on && matches!(l.kind, LightKind::Directional { .. })).into_iter().collect()
    } else {
        asked
    };
    let scene_r = (bounds.size().length() * 0.5).max(1e-3);
    let mut next_local = 0usize;
    for i in casters {
        let l = &lights[i];
        match l.kind {
            LightKind::Directional { direction } if plan.sun.is_none() => {
                plan.sun = Some((i, l.shadow_intensity));
                plan.cascades = budget.cascades.clamp(1, SUN_CASCADES_MAX as u32);
                let (near, far) = camera.near_far();
                let far = far.min(camera.distance + camera.scene_radius * 1.5).max(near * 2.0);
                let splits = practical_splits(near, far, plan.cascades as usize);
                for c in 0..plan.cascades as usize {
                    let start = if c == 0 { near } else { splits[c - 1] };
                    let corners = slice_corners(camera, aspect, start, splits[c]);
                    let (m, texel) = stable_cascade(&corners, direction, bounds, budget.sun_shadow_size);
                    plan.mats[c] = m;
                    plan.cascade_far[c] = splits[c];
                    plan.cascade_texel[c] = texel;
                    plan.passes.push(c);
                }
            }
            LightKind::Spot { location, direction, cut_off_angle, radius, .. } => {
                if next_local + 1 > budget.local_shadow_layers as usize {
                    plan.dropped.push(i);
                    continue;
                }
                let m = SUN_CASCADES_MAX + next_local;
                let fov = (2.0 * cut_off_angle).clamp(0.05, 3.0);
                plan.mats[m] = perspective_from(location, direction, fov, radius.min(scene_r * 4.0), scene_r);
                plan.local.push((i, m));
                plan.passes.push(m);
                next_local += 1;
            }
            LightKind::Point { location, radius, .. } => {
                if next_local + 6 > budget.local_shadow_layers as usize {
                    plan.dropped.push(i);
                    continue;
                }
                let first = SUN_CASCADES_MAX + next_local;
                // 92° rather than 90°: a little overlap, so PCF taps at a
                // face's edge still land inside that face's map.
                let fov = 92f32.to_radians();
                for (f, dir) in CUBE_FACES.iter().enumerate() {
                    plan.mats[first + f] = perspective_from(location, *dir, fov, radius.min(scene_r * 4.0), scene_r);
                    plan.passes.push(first + f);
                }
                plan.local.push((i, first));
                next_local += 6;
            }
            _ => plan.dropped.push(i),
        }
    }
    for &i in &plan.dropped {
        log::warn!("light {i} asked for a shadow but the budget has no layers left for it (or it is a second sun)");
    }
    plan
}

/// The order the shader picks faces in: +X −X +Y −Y +Z −Z.
const CUBE_FACES: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];

fn up_for(dir: Vec3) -> Vec3 {
    if dir.normalize_or(Vec3::NEG_Y).y.abs() > 0.99 { Vec3::Z } else { Vec3::Y }
}

fn perspective_from(eye: Vec3, dir: Vec3, fov: f32, reach: f32, scene_r: f32) -> Mat4 {
    let dir = dir.normalize_or(Vec3::NEG_Z);
    let near = (scene_r * 0.002).max(0.02);
    let far = reach.max(near * 10.0);
    let view = glam::camera::rh::view::look_at_mat4(eye, eye + dir, up_for(dir));
    glam::camera::rh::proj::directx::perspective(fov, 1.0, near, far) * view
}

/// Where cascade i ends: λ·log + (1−λ)·uniform (GPU Gems 3, ch. 10).
pub fn practical_splits(near: f32, far: f32, n: usize) -> [f32; 4] {
    let mut out = [far; 4];
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        let t = (i + 1) as f32 / n as f32;
        let log = near * (far / near).powf(t);
        let uni = near + (far - near) * t;
        *slot = SPLIT_LAMBDA * log + (1.0 - SPLIT_LAMBDA) * uni;
    }
    out
}

/// The 8 world-space corners of the view frustum between two depths.
pub fn slice_corners(camera: &OrbitCamera, aspect: f32, start: f32, end: f32) -> [Vec3; 8] {
    let inv = camera.view().inverse();
    let (right, up, back) = (inv.x_axis.xyz(), inv.y_axis.xyz(), inv.z_axis.xyz());
    let eye = camera.eye();
    let t = (camera.fov_y(aspect) * 0.5).tan();
    let mut out = [Vec3::ZERO; 8];
    for (k, d) in [start, end].into_iter().enumerate() {
        let (h, w) = (d * t, d * t * aspect);
        let c = eye - back * d;
        for (j, (sx, sy)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
            out[k * 4 + j] = c + right * (w * sx) + up * (h * sy);
        }
    }
    out
}

/// A shimmer-free orthographic box around `corners`, looking along `dir`.
/// Returns the world → clip matrix and the size of one texel in world units.
pub fn stable_cascade(corners: &[Vec3; 8], dir: Vec3, bounds: &Aabb, size: u32) -> (Mat4, f32) {
    let dir = dir.normalize_or(Vec3::NEG_Y);
    let center = corners.iter().copied().sum::<Vec3>() / 8.0;
    let r = corners.iter().map(|c| c.distance(center)).fold(0.0, f32::max);
    // Round the radius up to 1/16 unit: it then doesn't flicker as the camera turns.
    let r = ((r * 16.0).ceil() / 16.0).max(1e-3);
    let texel = 2.0 * r / size as f32;
    // A light space with no translation and a fixed up: only the box moves.
    let rot = glam::camera::rh::view::look_at_mat4(Vec3::ZERO, dir, up_for(dir));
    let c = rot.transform_point3(center);
    let (cx, cy) = ((c.x / texel).floor() * texel, (c.y / texel).floor() * texel);
    // Depth: everything in the scene that could cast into this slice.
    let mut zs: Vec<f32> = corners.iter().map(|p| rot.transform_point3(*p).z).collect();
    if !bounds.is_empty() {
        for i in 0..8 {
            let p = Vec3::new(
                if i & 1 == 0 { bounds.min.x } else { bounds.max.x },
                if i & 2 == 0 { bounds.min.y } else { bounds.max.y },
                if i & 4 == 0 { bounds.min.z } else { bounds.max.z },
            );
            zs.push(rot.transform_point3(p).z);
        }
    }
    let (zmin, zmax) = zs.iter().fold((f32::MAX, f32::MIN), |(a, b), &z| (a.min(z), b.max(z)));
    let pad = (zmax - zmin) * 0.01 + 1e-3;
    let proj = glam::camera::rh::proj::directx::orthographic(cx - r, cx + r, cy - r, cy + r, -zmax - pad, -zmin + pad);
    (proj * rot, texel)
}

/// GPU side: the two depth arrays, the sampler, the depth-only pipeline.
#[derive(Debug)]
pub struct ShadowMaps {
    pub sun_size: u32,
    pub local_size: u32,
    pub sun_array: wgpu::TextureView,
    pub local_array: wgpu::TextureView,
    sun_layers: Vec<wgpu::TextureView>,
    local_layers: Vec<wgpu::TextureView>,
    pub sampler: wgpu::Sampler,
    frame_bind_group: wgpu::BindGroup,
    empty_bind_group: wgpu::BindGroup,
    pass_bind_group: wgpu::BindGroup,
    layout: wgpu::PipelineLayout,
    pub pipeline: Option<wgpu::RenderPipeline>,
}

impl ShadowMaps {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        budget: &Budget,
        frame_buffer: &wgpu::Buffer,
        draw_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let array = |label, size: u32, layers: u32| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.max(1), height: size.max(1), depth_or_array_layers: layers.max(1) },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let whole = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            let layers = (0..layers)
                .map(|l| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: l,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect::<Vec<_>>();
            (whole, layers)
        };
        let (sun_array, sun_layers) = array("sun shadows", budget.sun_shadow_size, budget.cascades);
        let (local_array, local_layers) =
            array("spot/point shadows", budget.local_shadow_size, budget.local_shadow_layers);

        let uniform = |binding, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: None,
            },
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow frame"),
            entries: &[uniform(0, false)],
        });
        let empty_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("empty"), entries: &[] });
        let pass_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow pass"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });
        // Slot m holds the number m: written once, never again.
        let mut bytes = vec![0u8; PASS_STRIDE as usize * MAX_SHADOW_MATS];
        for m in 0..MAX_SHADOW_MATS {
            bytes[m * PASS_STRIDE as usize..][..4].copy_from_slice(&(m as u32).to_le_bytes());
        }
        let pass_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow pass index"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pass_buffer, 0, &bytes);
        Self {
            sun_size: budget.sun_shadow_size,
            local_size: budget.local_shadow_size,
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
            empty_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("empty"),
                layout: &empty_layout,
                entries: &[],
            }),
            pass_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow pass"),
                layout: &pass_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &pass_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                }],
            }),
            layout: device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[Some(&frame_layout), Some(draw_layout), Some(&empty_layout), Some(&pass_layout)],
                immediate_size: 0,
            }),
            sun_array,
            local_array,
            sun_layers,
            local_layers,
            pipeline: None,
        }
    }

    pub fn pipeline(
        &self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        cache: Option<&wgpu::PipelineCache>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                // Same vertex buffers as the scene; only the position is read.
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<x3d_scene::Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            // No culling: thin single-sided X3D geometry should still cast.
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                // Slope-scaled bias: surfaces seen edge-on by the light get more.
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache,
        })
    }

    /// One depth pass per planned matrix. `casters` yields
    /// (draw-uniform offset, vertex buffer, index buffer, index count).
    pub fn record<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &ShadowPlan,
        draws: &wgpu::BindGroup,
        casters: &[(u32, &'a wgpu::Buffer, &'a wgpu::Buffer, u32)],
        timing: &(Option<wgpu::RenderPassTimestampWrites<'_>>, Option<wgpu::RenderPassTimestampWrites<'_>>),
    ) {
        let pipeline = self.pipeline.as_ref().expect("shadow pipeline built");
        let last = plan.passes.len().saturating_sub(1);
        for (k, &m) in plan.passes.iter().enumerate() {
            let view =
                if m < SUN_CASCADES_MAX { &self.sun_layers[m] } else { &self.local_layers[m - SUN_CASCADES_MAX] };
            let timestamp_writes = match k {
                0 if last == 0 => timing.0.clone().map(|mut t| {
                    t.end_of_pass_write_index = timing.1.as_ref().and_then(|e| e.end_of_pass_write_index);
                    t
                }),
                0 => timing.0.clone(),
                k if k == last => timing.1.clone(),
                _ => None,
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(2, &self.empty_bind_group, &[]);
            pass.set_bind_group(3, &self.pass_bind_group, &[(m as u64 * PASS_STRIDE) as u32]);
            for &(offset, vertices, indices, count) in casters {
                pass.set_bind_group(1, draws, &[offset]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_bounds() -> Aabb {
        Aabb { min: Vec3::splat(-10.0), max: Vec3::splat(10.0) }
    }

    #[test]
    fn splits_grow_and_end_at_far() {
        let s = practical_splits(0.1, 100.0, 4);
        assert!(s.windows(2).all(|w| w[0] < w[1]), "{s:?}");
        assert!((s[3] - 100.0).abs() < 1e-3);
        assert!(s[0] < 25.0, "the first cascade is thinner than a uniform quarter: {s:?}");
    }

    #[test]
    fn cascade_boxes_move_in_whole_texels() {
        // Slide a slice of the view a little; the box must move by whole
        // texels (or not at all) and keep its size: that is what stops shimmer.
        let dir = Vec3::new(1.0, -2.0, 0.5);
        let corners = |shift: f32| {
            let mut c = [Vec3::ZERO; 8];
            for (i, p) in c.iter_mut().enumerate() {
                *p = Vec3::new((i & 1) as f32 * 4.0 + shift, (i >> 1 & 1) as f32 * 3.0, (i >> 2) as f32 * 5.0);
            }
            c
        };
        let (a, ta) = stable_cascade(&corners(0.0), dir, &box_bounds(), 2048);
        for shift in [0.001, 0.013, 0.37, 1.0] {
            let (b, tb) = stable_cascade(&corners(shift), dir, &box_bounds(), 2048);
            assert_eq!(ta, tb, "same texel size");
            // A fixed world point lands on the same sub-texel position.
            let p = Vec3::new(1.234, 0.5, 2.0);
            let (pa, pb) = (a.project_point3(p), b.project_point3(p));
            let texels = |v: f32| v * 1024.0; // NDC −1..1 over 2048 texels
            let dx = texels(pa.x) - texels(pb.x);
            let dy = texels(pa.y) - texels(pb.y);
            assert!((dx - dx.round()).abs() < 1e-2 && (dy - dy.round()).abs() < 1e-2, "shift {shift}: {dx}, {dy}");
        }
    }

    #[test]
    fn the_whole_slice_is_inside_its_cascade() {
        let corners = [
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-3.0, -3.0, -6.0),
            Vec3::new(3.0, -3.0, -6.0),
            Vec3::new(3.0, 3.0, -6.0),
            Vec3::new(-3.0, 3.0, -6.0),
        ];
        let (m, _) = stable_cascade(&corners, Vec3::new(0.3, -1.0, 0.2), &box_bounds(), 1024);
        for c in corners {
            let p = m.project_point3(c);
            assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z), "{c} → {p}");
        }
    }
}
