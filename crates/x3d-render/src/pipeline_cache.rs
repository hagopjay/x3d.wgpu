//! A pipeline cache on disk: compiled pipelines survive between runs.
//!
//! 🎨 Creating a pipeline makes the driver compile SPIR-V into GPU machine
//! code, which can take milliseconds per pipeline (seconds on a big scene).
//! Vulkan can hand that work back as an opaque blob; next run we give it
//! back and the driver skips the compile. wgpu wraps this as
//! [`wgpu::PipelineCache`] (feature `PIPELINE_CACHE`, Vulkan only today).
//!
//! The file lives in `$X3D_PIPELINE_CACHE_DIR`, else `$XDG_CACHE_HOME/x3d-wgpu`,
//! else `~/.cache/x3d-wgpu`, named by wgpu's key for this adapter + driver.
//! `X3D_PIPELINE_CACHE=off` disables it.

// 🦀 The one `unsafe` in the project. `create_pipeline_cache` is unsafe
// because the driver trusts the blob. We only pass back bytes that
// `PipelineCache::get_data` produced for the *same* `pipeline_cache_key`
// (it is the file name), and `fallback: true` makes wgpu discard data whose
// header doesn't match this adapter, driver or wgpu version.
#![allow(unsafe_code)]

use std::path::PathBuf;

pub fn dir() -> Option<PathBuf> {
    if std::env::var("X3D_PIPELINE_CACHE").is_ok_and(|v| v == "off") {
        return None;
    }
    std::env::var_os("X3D_PIPELINE_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(|d| PathBuf::from(d).join("x3d-wgpu")))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/x3d-wgpu")))
}

/// Open (or start) the cache for this adapter. Returns the cache, its file
/// and how many bytes were loaded.
pub fn open(device: &wgpu::Device, info: &wgpu::AdapterInfo) -> Option<(wgpu::PipelineCache, PathBuf, usize)> {
    if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
        return None;
    }
    let file = dir()?.join(format!("{}.bin", wgpu::util::pipeline_cache_key(info)?));
    let data = std::fs::read(&file).ok();
    let loaded = data.as_ref().map_or(0, Vec::len);
    // SAFETY: see the module comment: `data` is our own earlier `get_data`
    // output for this exact cache key, and `fallback` rejects mismatches.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("x3d pipelines"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    log::info!("pipeline cache {}: {loaded} bytes loaded", file.display());
    Some((cache, file, loaded))
}

/// Write the cache back (atomically: temp file + rename). Returns the size.
pub fn save(cache: &wgpu::PipelineCache, file: &std::path::Path) -> std::io::Result<usize> {
    let Some(data) = cache.get_data() else { return Ok(0) };
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, file)?;
    log::info!("pipeline cache {}: {} bytes saved", file.display(), data.len());
    Ok(data.len())
}
