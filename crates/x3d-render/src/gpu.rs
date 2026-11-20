//! Getting a GPU: Instance → Adapter → Device + Queue.
//!
//! 🎨 **The wgpu object model in one breath**
//! - [`wgpu::Instance`]: the entry point. Knows which *backends* (Vulkan,
//!   Metal, DX12, GL) to try.
//! - [`wgpu::Adapter`]: one physical GPU (or a software one like
//!   *llvmpipe/lavapipe*). You can ask what it supports.
//! - [`wgpu::Device`]: your logical connection to it. Creates buffers,
//!   textures, pipelines.
//! - [`wgpu::Queue`]: where finished command buffers go to run.
//!
//! On Linux, `Backend::Vulkan` is a genuine Vulkan renderer: wgpu turns our
//! WGSL into SPIR-V (via `naga`) and records Vulkan command buffers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::budget::{Budget, Profile};

/// Which graphics API to use.
///
/// 🦀 `clap::ValueEnum` lets the CLI accept `--backend vulkan` and lists
/// the choices in `--help` automatically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Backend {
    /// The best available for this OS.
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl Backend {
    pub fn to_wgpu(self) -> wgpu::Backends {
        match self {
            Self::Auto => wgpu::Backends::PRIMARY,
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Metal => wgpu::Backends::METAL,
            Self::Dx12 => wgpu::Backends::DX12,
            Self::Gl => wgpu::Backends::GL,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error(
        "no {backend:?} GPU adapter found ({source}). On Linux without a GPU, install a software Vulkan driver: `apt install mesa-vulkan-drivers`"
    )]
    NoAdapter {
        backend: Backend,
        #[source]
        source: wgpu::RequestAdapterError,
    },
    #[error("the adapter refused to create a device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
}

/// Everything needed to talk to one GPU.
#[derive(Debug, Clone)]
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// What this GPU can afford, and why (see `budget.rs`).
    pub budget: Budget,
    /// Set by wgpu's device-lost callback (driver reset, GPU unplugged,
    /// `device.destroy()`). The viewer polls it and rebuilds everything.
    pub lost: Arc<AtomicBool>,
}

pub fn make_instance(backend: Backend) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = backend.to_wgpu();
    // 🎨 X3D_VVL=1|require asks for the Vulkan validation layer (see `validation`).
    if crate::validation::mode() != crate::validation::Mode::Default {
        desc.flags |= wgpu::InstanceFlags::VALIDATION;
    }
    crate::validation::note_instance(desc.flags.with_env().contains(wgpu::InstanceFlags::VALIDATION));
    wgpu::Instance::new(desc)
}

/// Optional features we use when the adapter has them.
const WANTED: wgpu::Features = wgpu::Features::POLYGON_MODE_LINE
    .union(wgpu::Features::TIMESTAMP_QUERY)
    .union(wgpu::Features::PIPELINE_CACHE)
    .union(wgpu::Features::INDIRECT_FIRST_INSTANCE);

impl Gpu {
    /// A GPU for off-screen work (no window).
    pub fn headless(backend: Backend) -> Result<Self, GpuError> {
        Self::headless_with(backend, Profile::Auto)
    }

    pub fn headless_with(backend: Backend, profile: Profile) -> Result<Self, GpuError> {
        // 🦀 wgpu's setup is `async` (it must be on the web). On native we
        // simply block on the future with `pollster`.
        pollster::block_on(Self::new(make_instance(backend), backend, profile, None))
    }

    /// A GPU that can present to `surface` (if given).
    pub async fn new(
        instance: wgpu::Instance,
        backend: Backend,
        profile: Profile,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|source| GpuError::NoAdapter { backend, source })?;
        log::info!("adapter: {}", describe(&adapter.get_info()));

        // 🎨 Limits-aware: ask for what the adapter *has* (capped by our
        // profile), never for a fixed list it might not meet.
        let (features, limits) = match profile {
            Profile::Auto => (adapter.features() & WANTED, adapter.limits()),
            Profile::Weak => {
                (wgpu::Features::empty(), wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()))
            }
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("x3d device"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            })
            .await?;
        log::info!("device features: {features:?}");

        // By default wgpu *panics* on a validation error. A tinkering tool
        // should log instead (and the shader hot-reload path catches its own).
        device.on_uncaptured_error(Arc::new(|e| log::error!("wgpu validation error (reported, not fatal): {e}")));
        let lost = Arc::new(AtomicBool::new(false));
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            log::error!("GPU device lost ({reason:?}): {message}");
            flag.store(true, Ordering::SeqCst);
        });
        let budget = Budget::decide(&adapter, &device, profile);
        for d in &budget.decisions {
            log::info!("budget: {} = {} ({})", d.what, d.value, d.source);
        }
        Ok(Self { instance, adapter, device, queue, budget, lost })
    }

    /// Has the device been lost? Everything made from it is now useless.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    pub fn supports_wireframe(&self) -> bool {
        self.device.features().contains(wgpu::Features::POLYGON_MODE_LINE)
    }

    /// One line for logs: `llvmpipe (LLVM 20.1.2, 256 bits) — Vulkan, Cpu`.
    pub fn describe(&self) -> String {
        describe(&self.adapter.get_info())
    }
}

pub fn describe(info: &wgpu::AdapterInfo) -> String {
    format!("{} — {:?}, {:?}, driver {} {}", info.name, info.backend, info.device_type, info.driver, info.driver_info)
}

/// Every adapter wgpu can see (`x3d-view adapters`).
pub fn list_adapters(backend: Backend) -> Vec<wgpu::AdapterInfo> {
    let instance = make_instance(backend);
    pollster::block_on(instance.enumerate_adapters(backend.to_wgpu())).into_iter().map(|a| a.get_info()).collect()
}
