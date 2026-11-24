//! Textures: decode image files, build mipmaps, upload them, and make one
//! bind group per distinct *set* of material textures.
//!
//! 🎨 **Mipmaps** are pre-shrunk copies of a texture (½, ¼, … down to 1×1).
//! When a textured surface is far away, one screen pixel covers many
//! texels; sampling the matching smaller copy averages them instead of
//! picking one at random, which is what makes distant sand shimmer.
//! wgpu doesn't generate mipmaps for you, so we do it on the CPU here
//! (generating them on the GPU, in a compute shader, is a LEARNING.md quest).
//!
//! 🎨 **Anisotropic filtering** (`anisotropy_clamp: 16`) samples along the
//! direction a texture is stretched, so ground seen at a grazing angle
//! stays sharp. Giza's terrain asks for it (`anisotropicDegree='16'`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use x3d_scene::Textures;

use crate::uniforms::texture_bits;

#[derive(Debug)]
pub struct GpuTexture {
    pub view: wgpu::TextureView,
    /// Any texel less than fully opaque: the surface must be drawn as transparent.
    pub has_alpha: bool,
    pub bytes: u64,
}

/// One material's textures, ready to bind at group 2.
#[derive(Debug)]
pub struct MaterialTextures {
    pub bind_group: wgpu::BindGroup,
    /// Which slots hold a real texture (see [`texture_bits`]).
    pub bits: u32,
    pub has_alpha: bool,
}

#[derive(Debug)]
pub struct TextureCache {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// For a base texture that asks for sharp texels (PixelTexture): nearest when magnified.
    sampler_nearest: wgpu::Sampler,
    /// Bound to every unused slot: shaders can sample all five unconditionally.
    white: Arc<GpuTexture>,
    /// 🦀 `Option` remembers failures too, so a missing file warns once.
    images: HashMap<PathBuf, Option<Arc<GpuTexture>>>,
    sets: HashMap<Textures, Arc<MaterialTextures>>,
}

impl TextureCache {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
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
        let mut entries: Vec<_> = (0..5).map(texture_entry).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material textures layout"),
            entries: &entries,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16, // needs all three filters Linear
            ..Default::default()
        });
        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material sampler (nearest)"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let white = Arc::new(upload(device, queue, image::RgbaImage::from_pixel(1, 1, image::Rgba([255; 4])), "white"));
        Self {
            device: device.clone(),
            queue: queue.clone(),
            layout,
            sampler,
            sampler_nearest,
            white,
            images: HashMap::new(),
            sets: HashMap::new(),
        }
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Total GPU bytes of the loaded images (mipmaps included).
    pub fn bytes(&self) -> u64 {
        self.images.values().flatten().map(|t| t.bytes).sum()
    }

    /// The bind group for a material's textures (built once per distinct set).
    pub fn material(&mut self, textures: &Textures) -> Arc<MaterialTextures> {
        if let Some(set) = self.sets.get(textures) {
            return Arc::clone(set);
        }
        let mut bits = 0;
        let mut has_alpha = false;
        let mut views = Vec::with_capacity(5);
        let flags = [
            texture_bits::BASE,
            texture_bits::NORMAL,
            texture_bits::METAL_ROUGH,
            texture_bits::OCCLUSION,
            texture_bits::EMISSIVE,
        ];
        for (slot, bit) in textures.slots().into_iter().zip(flags) {
            let loaded = slot.and_then(|t| self.image(t));
            if let Some(t) = &loaded {
                bits |= bit;
                // Only the base colour's alpha makes a surface see-through.
                has_alpha |= bit == texture_bits::BASE && t.has_alpha;
            }
            views.push(loaded.unwrap_or_else(|| Arc::clone(&self.white)));
        }
        let mut entries: Vec<wgpu::BindGroupEntry> = views
            .iter()
            .enumerate()
            .map(|(i, t)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(&t.view),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(if textures.base.as_ref().is_some_and(|t| t.nearest) {
                &self.sampler_nearest
            } else {
                &self.sampler
            }),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material textures"),
            layout: &self.layout,
            entries: &entries,
        });
        let set = Arc::new(MaterialTextures { bind_group, bits, has_alpha });
        self.sets.insert(textures.clone(), Arc::clone(&set));
        set
    }

    fn image(&mut self, t: &x3d_scene::TextureRef) -> Option<Arc<GpuTexture>> {
        let path = t.path.as_path();
        if let Some(cached) = self.images.get(path) {
            return cached.clone();
        }
        // `data:` urls arrive decoded; files are read here.
        let decoded = match &t.inline {
            Some(bytes) => image::load_from_memory(bytes),
            None => image::open(path),
        };
        let loaded = match decoded {
            Ok(img) => {
                let label = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                Some(Arc::new(upload(&self.device, &self.queue, img.to_rgba8(), &label)))
            }
            Err(e) => {
                log::warn!("texture {}: {e}", path.display());
                None
            }
        };
        self.images.insert(path.to_owned(), loaded.clone());
        loaded
    }
}

/// Upload an image with a full mip chain.
fn upload(device: &wgpu::Device, queue: &wgpu::Queue, img: image::RgbaImage, label: &str) -> GpuTexture {
    let (w, h) = img.dimensions();
    // 🦀 `ilog2` on integers: a 1024-wide texture has 11 levels (1024 … 1).
    let levels = w.max(h).max(1).ilog2() + 1;
    let has_alpha = img.pixels().any(|p| p.0[3] < 250);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Plain Unorm, not *Srgb*: the shader decides how to read each slot
        // (classic X3D uses colours as they are; PBR decodes base/emissive).
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut bytes = 0u64;
    let mut level_img = img;
    for level in 0..levels {
        let (lw, lh) = level_img.dimensions();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level_img,
            // write_texture has no 256-byte row rule (only buffer copies do).
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(lh) },
            wgpu::Extent3d { width: lw, height: lh, depth_or_array_layers: 1 },
        );
        bytes += u64::from(4 * lw * lh);
        if level + 1 < levels {
            let (nw, nh) = ((lw / 2).max(1), (lh / 2).max(1));
            level_img = image::imageops::resize(&level_img, nw, nh, image::imageops::FilterType::Triangle);
        }
    }
    GpuTexture { view: texture.create_view(&Default::default()), has_alpha, bytes }
}
