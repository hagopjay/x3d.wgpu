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
}

pub fn make_instance(backend: Backend) -> wgpu::Instance {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = backend.to_wgpu();
    wgpu::Instance::new(desc)
}

impl Gpu {
    /// A GPU for off-screen work (no window).
    pub fn headless(backend: Backend) -> Result<Self, GpuError> {
        // 🦀 wgpu's setup is `async` (it must be on the web). On native we
        // simply block on the future with `pollster`.
        pollster::block_on(Self::new(make_instance(backend), backend, None))
    }

    /// A GPU that can present to `surface` (if given).
    pub async fn new(
        instance: wgpu::Instance,
        backend: Backend,
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

        // Ask only for optional features the adapter really has.
        let wanted = wgpu::Features::POLYGON_MODE_LINE;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("x3d device"),
                required_features: adapter.features() & wanted,
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await?;

        // By default wgpu *panics* on a validation error. A tinkering tool
        // should log instead (and the shader hot-reload path catches its own).
        device.on_uncaptured_error(Arc::new(|e| log::error!("wgpu: {e}")));
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    pub fn supports_wireframe(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
    }

    /// One line for logs: `llvmpipe (LLVM 20.1.2, 256 bits) — Vulkan, Cpu`.
    pub fn describe(&self) -> String {
        describe(&self.adapter.get_info())
    }
}

pub fn describe(info: &wgpu::AdapterInfo) -> String {
    format!(
        "{} — {:?}, {:?}, driver {} {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    )
}

/// Every adapter wgpu can see (`x3d-view adapters`).
pub fn list_adapters(backend: Backend) -> Vec<wgpu::AdapterInfo> {
    let instance = make_instance(backend);
    pollster::block_on(instance.enumerate_adapters(backend.to_wgpu()))
        .into_iter()
        .map(|a| a.get_info())
        .collect()
}
