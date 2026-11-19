//! Frustum culling, on the CPU or the GPU.
//!
//! 🎨 Everything outside the camera's view volume (a truncated pyramid,
//! the *frustum*) can be skipped. Each draw keeps a bounding sphere; a
//! sphere is outside if it lies entirely behind one of the six planes.
//!
//! - **CPU**: test spheres before recording; culled draws are never issued.
//! - **GPU**: a compute shader (`cull.wgsl`) writes each draw's
//!   *indirect* arguments; the CPU issues every draw with
//!   `draw_indexed_indirect`, and the GPU skips the culled ones. That's the
//!   first step of GPU-driven rendering: the next is merging meshes into one
//!   buffer and replacing the per-draw calls with one `multi_draw_indexed_indirect`.

use glam::{Mat4, Vec3, Vec4};

/// Which culling to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Cull {
    Off,
    /// Bounding spheres against the frustum on the CPU (default).
    #[default]
    Cpu,
    /// A compute pass writes indirect-draw arguments (falls back to CPU
    /// where the budget has no compute).
    Gpu,
}

/// The six planes of `view_proj` (Gribb & Hartmann), normalised, for wgpu's
/// 0..1 depth: left, right, bottom, top, near, far.
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let r = |i: usize| view_proj.row(i);
    let planes = [r(3) + r(0), r(3) - r(0), r(3) + r(1), r(3) - r(1), r(2), r(3) - r(2)];
    planes.map(|p| p / p.truncate().length().max(1e-12))
}

pub fn sphere_visible(planes: &[Vec4; 6], center: Vec3, radius: f32) -> bool {
    planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius)
}

/// GPU culling resources, sized per scene.
#[derive(Debug)]
pub struct GpuCull {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    counter: wgpu::Buffer,
    readback: wgpu::Buffer,
    pub spheres: Option<wgpu::Buffer>,
    pub args: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniform {
    planes: [[f32; 4]; 6],
    count: [u32; 4],
}

impl GpuCull {
    pub fn new(device: &wgpu::Device, cache: Option<&wgpu::PipelineCache>) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/cull.wgsl"));
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cull"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cull"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull_main"),
            compilation_options: Default::default(),
            cache,
        });
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        Self {
            pipeline,
            layout,
            uniform: buffer(
                "cull",
                size_of::<CullUniform>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            counter: buffer(
                "cull visible",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer("cull readback", 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            spheres: None,
            args: None,
            bind_group: None,
            count: 0,
        }
    }

    /// Per scene: `index_counts[i]` for draw i (0 for non-indexed draws).
    pub fn set_scene(&mut self, device: &wgpu::Device, index_counts: &[u32]) {
        use wgpu::util::DeviceExt;
        self.count = index_counts.len() as u32;
        let n = index_counts.len().max(1);
        let mut args = vec![0u32; n * 5];
        for (i, &c) in index_counts.iter().enumerate() {
            args[i * 5] = c;
            args[i * 5 + 1] = 1;
        }
        let args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indirect args"),
            contents: bytemuck::cast_slice(&args),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        });
        let spheres = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw spheres"),
            size: (n * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cull"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spheres.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: args.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.counter.as_entire_binding() },
            ],
        }));
        self.spheres = Some(spheres);
        self.args = Some(args);
    }

    /// Record the culling dispatch for this frame's planes.
    pub fn record(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        planes: &[Vec4; 6],
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let Some(group) = &self.bind_group else { return };
        let uniform = CullUniform { planes: planes.map(|p| p.to_array()), count: [self.count, 0, 0, 0] };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        queue.write_buffer(&self.counter, 0, &0u32.to_le_bytes());
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cull"), timestamp_writes });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64).max(1), 1, 1);
        drop(pass);
        encoder.copy_buffer_to_buffer(&self.counter, 0, &self.readback, 0, 4);
    }

    /// How many draws the GPU kept last frame (blocks: tests and `bench` only).
    pub fn visible_blocking(&self, device: &wgpu::Device) -> Option<u32> {
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let n = slice.get_mapped_range().ok().map(|v| u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
        self.readback.unmap();
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spheres_in_front_pass_and_behind_fail() {
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(1.0, 1.0, 0.1, 100.0);
        let p = frustum_planes(proj * view);
        assert!(sphere_visible(&p, Vec3::ZERO, 1.0));
        assert!(!sphere_visible(&p, Vec3::new(0.0, 0.0, 10.0), 1.0), "behind the camera");
        assert!(!sphere_visible(&p, Vec3::new(50.0, 0.0, 0.0), 1.0), "far to the right");
        assert!(sphere_visible(&p, Vec3::new(3.2, 0.0, 0.0), 1.0), "straddles the right plane");
        assert!(!sphere_visible(&p, Vec3::new(0.0, 0.0, -200.0), 1.0), "beyond far");
    }
}
