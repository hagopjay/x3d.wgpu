//! Rendering without a window, and reading pixels back from the GPU.
//!
//! 🎨 **The readback dance.** GPU memory isn't directly readable. We:
//! 1. render into a texture with `COPY_SRC` usage,
//! 2. copy it into a buffer with `MAP_READ` usage,
//! 3. *map* the buffer (an async request), wait for the GPU with `poll`,
//! 4. copy rows out — skipping padding, because each row in the buffer must
//!    start at a multiple of 256 bytes (`COPY_BYTES_PER_ROW_ALIGNMENT`).

use image::RgbaImage;
use x3d_scene::Scene;

use crate::camera::OrbitCamera;
use crate::gpu::Gpu;
use crate::renderer::{DebugView, RenderSettings, Renderer};

#[derive(Debug, thiserror::Error)]
pub enum ReadbackError {
    #[error("{0:?} can't be read back as 8-bit RGBA")]
    Format(wgpu::TextureFormat),
    #[error("mapping the readback buffer failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("waiting for the GPU failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("the mapping callback never ran")]
    NoCallback,
}

/// A texture you can render into and then read back.
#[derive(Debug)]
pub struct Offscreen {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl Offscreen {
    pub fn new(device: &wgpu::Device, size: (u32, u32), format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view, size }
    }

    pub fn read(&self, gpu: &Gpu) -> Result<RgbaImage, ReadbackError> {
        read_texture(gpu, &self.texture)
    }
}

/// Copy an 8-bit RGBA/BGRA texture into an `image::RgbaImage`.
pub fn read_texture(gpu: &Gpu, texture: &wgpu::Texture) -> Result<RgbaImage, ReadbackError> {
    use wgpu::TextureFormat as F;
    let format = texture.format();
    let bgra = match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => false,
        F::Bgra8Unorm | F::Bgra8UnormSrgb => true,
        other => return Err(ReadbackError::Format(other)),
    };
    let (w, h) = (texture.width(), texture.height());
    let unpadded = w * 4;
    let padded = unpadded.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(padded) * u64::from(h),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);

    // 🦀 A channel carries the callback's result back to this thread.
    let (tx, rx) = std::sync::mpsc::channel();
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv().map_err(|_| ReadbackError::NoCallback)??;

    let mut pixels = Vec::with_capacity((unpadded * h) as usize);
    {
        let data = slice.get_mapped_range().map_err(|_| ReadbackError::NoCallback)?;
        for row in data.chunks_exact(padded as usize) {
            pixels.extend_from_slice(&row[..unpadded as usize]);
        }
    } // the mapped view must be dropped before `unmap`
    buffer.unmap();
    if bgra {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    Ok(RgbaImage::from_raw(w, h, pixels).expect("buffer is exactly w×h×4"))
}

/// Options for [`render_scene`].
#[derive(Debug, Clone)]
pub struct Shot {
    pub size: (u32, u32),
    pub msaa: u32,
    pub settings: RenderSettings,
    /// Which `<Viewpoint>` to use; `None` = the first, or frame the scene.
    pub viewpoint: Option<usize>,
}

impl Default for Shot {
    fn default() -> Self {
        Self { size: (1280, 720), msaa: 4, settings: RenderSettings::default(), viewpoint: None }
    }
}

/// The starting camera for a scene: its chosen Viewpoint, or a framing of
/// its bounds when it has none.
pub fn camera_for(scene: &Scene, viewpoint: Option<usize>) -> OrbitCamera {
    let bounds = scene.bounds();
    match scene.viewpoints.get(viewpoint.unwrap_or(0)) {
        Some(vp) => OrbitCamera::from_viewpoint(vp, bounds),
        None => OrbitCamera::framing(bounds),
    }
}

/// Render a scene to an image in one call.
pub fn render_scene(gpu: &Gpu, scene: &Scene, shot: &Shot) -> Result<RgbaImage, ReadbackError> {
    let mut renderer = Renderer::new(gpu, wgpu::TextureFormat::Rgba8Unorm, shot.msaa);
    renderer.set_scene(scene);
    let camera = camera_for(scene, shot.viewpoint);
    let target = Offscreen::new(&gpu.device, shot.size, renderer.color_format());
    renderer.render(&target.view, shot.size, &camera, &shot.settings);
    target.read(gpu)
}

/// All seven debug views in a 4 × 2 grid — a quick way to *see* normals,
/// depth and winding problems at once.
pub fn render_sheet(gpu: &Gpu, scene: &Scene, shot: &Shot) -> Result<RgbaImage, ReadbackError> {
    let (w, h) = shot.size;
    let mut sheet = RgbaImage::new(w * 4, h * 2);
    let mut renderer = Renderer::new(gpu, wgpu::TextureFormat::Rgba8Unorm, shot.msaa);
    renderer.set_scene(scene);
    let camera = camera_for(scene, shot.viewpoint);
    let target = Offscreen::new(&gpu.device, shot.size, renderer.color_format());
    for (i, view) in DebugView::ALL.into_iter().enumerate() {
        let settings = RenderSettings { view, ..shot.settings };
        renderer.render(&target.view, shot.size, &camera, &settings);
        let tile = target.read(gpu)?;
        image::imageops::replace(&mut sheet, &tile, i64::from(w) * (i as i64 % 4), i64::from(h) * (i as i64 / 4));
    }
    Ok(sheet)
}
