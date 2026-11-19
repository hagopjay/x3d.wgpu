//! What this GPU can afford, decided once at start-up, with the reason for
//! every decision.
//!
//! 🎨 A renderer that asks for "the defaults" either refuses to start on a
//! weak GPU or wastes a strong one. Instead we read the adapter's *limits*
//! (numbers: biggest texture, array layers, workgroup size…), its optional
//! *features* (timestamps, pipeline caches…) and per-format capabilities,
//! and pick settings that fit. Every [`Decision`] names the limit, feature
//! or format query it came from, so `x3d-view adapters --budget` can print
//! a table where no number is unexplained.

use std::fmt;

/// How much to ask of the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Profile {
    /// Everything the adapter offers.
    #[default]
    Auto,
    /// Pretend to be a WebGL2-class GPU (`Limits::downlevel_webgl2_defaults`,
    /// no optional features): what you get where Chrome's WebGPU blocklists
    /// the GPU (e.g. Intel UHD 620) and only the WebGL path is left.
    Weak,
}

/// Rough class of the device, for the summary line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Full,
    Reduced,
    Minimal,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub what: &'static str,
    pub value: String,
    /// The limit, feature or query this came from, with its value.
    pub source: String,
}

/// The settings this GPU gets. `Renderer` clamps `RenderSettings` to these.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
    pub tier: Tier,
    /// Sun (cascade) shadow-map side, in texels.
    pub sun_shadow_size: u32,
    /// Number of sun cascades (1–4).
    pub cascades: u32,
    /// Side of each spot/point shadow layer (0 = no local-light shadows).
    pub local_shadow_size: u32,
    /// Local shadow layers: a spot light takes 1, a point light 6.
    pub local_shadow_layers: u32,
    /// Largest MSAA count both the colour and depth formats support.
    pub max_msaa: u32,
    /// Float render targets for tone mapping and bloom.
    pub hdr: bool,
    pub timestamps: bool,
    pub pipeline_cache: bool,
    /// Compute shaders + storage buffers + indirect draws (GPU frustum culling).
    pub gpu_cull: bool,
    pub decisions: Vec<Decision>,
}

impl Budget {
    /// Decide from what the *device* got (its limits and features are what
    /// we may use, which can be less than the adapter offers).
    pub fn decide(adapter: &wgpu::Adapter, device: &wgpu::Device, profile: Profile) -> Self {
        let info = adapter.get_info();
        let limits = device.limits();
        let features = device.features();
        let mut d = Vec::new();
        let mut note = |what, value: String, source: String| d.push(Decision { what, value, source });

        let tier = match (profile, info.device_type) {
            (Profile::Weak, _) => Tier::Minimal,
            (_, wgpu::DeviceType::Cpu) => Tier::Reduced,
            _ => Tier::Full,
        };
        note(
            "tier",
            format!("{tier:?}"),
            format!("profile {profile:?}, device_type {:?} ({})", info.device_type, info.name),
        );

        let max_2d = limits.max_texture_dimension_2d;
        let want_sun = match tier {
            Tier::Full => 4096,
            Tier::Reduced => 2048,
            Tier::Minimal => 1024,
        };
        let sun_shadow_size = want_sun.min(max_2d);
        note(
            "sun shadow map",
            format!("{sun_shadow_size}²"),
            format!("tier {tier:?} wants {want_sun}; limits.max_texture_dimension_2d = {max_2d}"),
        );

        let layers = limits.max_texture_array_layers;
        let cascades = match tier {
            Tier::Minimal => 1,
            _ => 4,
        }
        .min(layers);
        note("cascades", cascades.to_string(), format!("tier {tier:?}; limits.max_texture_array_layers = {layers}"));

        let (local_shadow_size, local_shadow_layers) = match tier {
            Tier::Full => (1024, 16),
            Tier::Reduced => (512, 16),
            Tier::Minimal => (0, 0),
        };
        let local_shadow_layers = local_shadow_layers.min(layers);
        note(
            "spot/point shadow layers",
            format!("{local_shadow_layers} × {local_shadow_size}²"),
            format!(
                "tier {tier:?} (a spot light takes 1 layer, a point light 6); limits.max_texture_array_layers = {layers}"
            ),
        );
        let mib = |side: u32, n: u32| (u64::from(side) * u64::from(side) * 4 * u64::from(n)) as f64 / 1048576.0;
        note(
            "shadow memory",
            format!("{:.0} MiB", mib(sun_shadow_size, cascades) + mib(local_shadow_size, local_shadow_layers)),
            "Depth32Float = 4 bytes per texel, sum over all layers".into(),
        );

        let msaa_of = |format: wgpu::TextureFormat| {
            let f = adapter.get_texture_format_features(format).flags;
            if f.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4) { 4 } else { 1 }
        };
        let max_msaa = msaa_of(wgpu::TextureFormat::Rgba8Unorm).min(msaa_of(wgpu::TextureFormat::Depth32Float));
        note(
            "max MSAA",
            max_msaa.to_string(),
            "texture-format features MULTISAMPLE_X4 of Rgba8Unorm and Depth32Float".into(),
        );

        let f16 = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let renderable = f16.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
            && f16.flags.contains(wgpu::TextureFormatFeatureFlags::FILTERABLE);
        let hdr = renderable && tier != Tier::Minimal;
        note(
            "HDR (tone map, bloom)",
            on(hdr),
            format!("Rgba16Float renderable+filterable = {renderable}; tier {tier:?} (Minimal emulates WebGL2: off)"),
        );

        let timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        note("GPU timestamps", on(timestamps), format!("feature TIMESTAMP_QUERY = {timestamps}"));
        let pipeline_cache = features.contains(wgpu::Features::PIPELINE_CACHE);
        note("pipeline cache", on(pipeline_cache), format!("feature PIPELINE_CACHE = {pipeline_cache}"));

        let compute = limits.max_compute_workgroup_size_x >= 64 && limits.max_storage_buffers_per_shader_stage >= 2;
        let gpu_cull = compute && tier != Tier::Minimal;
        note(
            "GPU frustum culling",
            on(gpu_cull),
            format!(
                "limits.max_compute_workgroup_size_x = {} (need 64), max_storage_buffers_per_shader_stage = {} (need 2)",
                limits.max_compute_workgroup_size_x, limits.max_storage_buffers_per_shader_stage
            ),
        );

        Self {
            tier,
            sun_shadow_size,
            cascades,
            local_shadow_size,
            local_shadow_layers,
            max_msaa,
            hdr,
            timestamps,
            pipeline_cache,
            gpu_cull,
            decisions: d,
        }
    }
}

fn on(b: bool) -> String {
    if b { "on" } else { "off" }.into()
}

impl fmt::Display for Budget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:<26} {:<12} source", "decision", "value")?;
        for d in &self.decisions {
            writeln!(f, "{:<26} {:<12} {}", d.what, d.value, d.source)?;
        }
        Ok(())
    }
}
