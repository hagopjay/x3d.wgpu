//! Quest 0: the smallest complete wgpu program.
//!
//! One triangle, rendered off-screen and saved as `hello_triangle.png`.
//! No helpers from this crate: every step is right here, in order.
//!
//! ```sh
//! cargo run -p x3d-render --example hello_triangle
//! ```
//!
//! Things to try (each is a one-line change):
//! - swap two vertices in `vs` → the triangle now faces away; add
//!   `cull_mode: Some(wgpu::Face::Back)` to the pipeline and watch it vanish.
//! - change `TriangleList` to `LineStrip`.
//! - make the fragment shader output `in.color * 0.5`.

use anyhow::{Context, Result};

const SIZE: u32 = 512;

// The vertices live *in the shader*: `vertex_index` picks one of three.
const SHADER: &str = r"
struct Out {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> Out {
    var corners = array(vec2f(0.0, 0.7), vec2f(-0.7, -0.6), vec2f(0.7, -0.6));
    var colors = array(vec3f(1.0, 0.2, 0.2), vec3f(0.2, 1.0, 0.3), vec3f(0.2, 0.4, 1.0));
    var out: Out;
    out.position = vec4f(corners[i], 0.0, 1.0);
    out.color = colors[i];
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4f {
    return vec4f(in.color, 1.0); // interpolated between the corners
}
";

fn main() -> Result<()> {
    // 1. Instance → adapter → device + queue.
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .context("no GPU adapter (on a headless Linux box: apt install mesa-vulkan-drivers)")?;
    println!("adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default()))?;

    // 2. A texture to draw into. RENDER_ATTACHMENT: we draw to it; COPY_SRC: we read it back.
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("canvas"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());

    // 3. Compile the shader and build a pipeline (shader + fixed-function state).
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hello"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hello pipeline"),
        layout: None, // let wgpu infer it: we have no buffers or textures to bind
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        multiview_mask: None,
        cache: None,
    });

    // 4. Record commands: one render pass that clears and draws 3 vertices.
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hello pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.08,
                        g: 0.08,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }

    // 5. Copy the texture into a buffer we can map. Rows must be 256-byte aligned
    //    (512 px × 4 bytes = 2048, already a multiple — try SIZE = 500 and see
    //    x3d_render::headless::read_texture for the padded version).
    let bytes_per_row = SIZE * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bytes_per_row * SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        texture.size(),
    );

    // 6. Submit, wait, read.
    queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely())?;
    let pixels = buffer.slice(..).get_mapped_range()?.to_vec();
    image::RgbaImage::from_raw(SIZE, SIZE, pixels)
        .context("wrong buffer size")?
        .save("hello_triangle.png")?;
    println!("wrote hello_triangle.png");
    Ok(())
}
