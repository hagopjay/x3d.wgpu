//! `x3d-view bench`: per-pass GPU times, CPU time and culling over N frames.
//!
//! Every column names where its number came from: GPU columns come from
//! timestamp queries (with the queue's tick period and the adapter); if the
//! adapter has no `TIMESTAMP_QUERY` they are `null`, never a CPU substitute.

use std::fmt;
// 🦀 wasm32-unknown-unknown has no clock in `std`; `web-time` reads the browser's.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use x3d_scene::Scene;

use crate::gpu::{Gpu, describe};
use crate::headless::{Offscreen, Shot, camera_for};
use crate::renderer::Renderer;
use crate::timing::{PASSES, PassTimes};

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub adapter: String,
    pub period_ns: Option<f32>,
    pub frames: usize,
    /// `Renderer::set_scene` (uploads; pipelines are built lazily).
    pub set_scene_ms: f64,
    /// The first frame: pipeline creation happens here (the cache's job).
    pub first_frame_ms: f64,
    pub pipeline_cache: Option<(String, usize)>,
    pub cpu_ms: Vec<f64>,
    pub gpu: Vec<PassTimes>,
    pub draw_calls: u32,
    pub culled: u32,
    pub gpu_visible: Option<u32>,
    pub shadow_passes: u32,
    pub draws_total: usize,
}

/// Render `frames` frames (orbiting 1° per frame so culling changes) and time them.
pub fn bench(gpu: &Gpu, scene: &Scene, shot: &Shot, frames: usize) -> BenchReport {
    let t0 = Instant::now();
    let mut renderer = Renderer::new(gpu, wgpu::TextureFormat::Rgba8Unorm, shot.msaa);
    renderer.set_scene(scene);
    let set_scene_ms = t0.elapsed().as_secs_f64() * 1e3;
    let target = Offscreen::new(&gpu.device, shot.size, renderer.color_format());
    let mut camera = camera_for(scene, shot.viewpoint);

    let t1 = Instant::now();
    renderer.render(&target.view, shot.size, &camera, &shot.settings);
    let _ = renderer.wait_gpu_times();
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let first_frame_ms = t1.elapsed().as_secs_f64() * 1e3;
    let pipeline_cache = renderer.save_pipeline_cache().map(|(f, n)| (f.display().to_string(), n));

    let (mut cpu_ms, mut times) = (Vec::new(), Vec::new());
    let mut last = None;
    for _ in 0..frames {
        camera.yaw += 1f32.to_radians();
        let stats = renderer.render(&target.view, shot.size, &camera, &shot.settings);
        cpu_ms.push(f64::from(stats.cpu_ms));
        if let Some(t) = renderer.wait_gpu_times() {
            times.push(t);
        }
        last = Some(stats);
    }
    let last = last.unwrap_or_default();
    BenchReport {
        adapter: describe(&gpu.adapter.get_info()),
        period_ns: gpu.budget.timestamps.then(|| gpu.queue.get_timestamp_period()),
        frames,
        set_scene_ms,
        first_frame_ms,
        pipeline_cache,
        cpu_ms,
        gpu: times,
        draw_calls: last.draw_calls,
        culled: last.culled,
        gpu_visible: renderer.gpu_visible(),
        shadow_passes: last.shadow_passes,
        draws_total: scene.draws.len(),
    }
}

fn stats(mut v: Vec<f64>) -> Option<(f64, f64, f64, f64)> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let pick = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    Some((v.iter().sum::<f64>() / v.len() as f64, pick(0.5), pick(0.95), v[v.len() - 1]))
}

impl fmt::Display for BenchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "adapter: {}", self.adapter)?;
        let source = match self.period_ns {
            Some(p) => format!("timestamp queries, {p} ns/tick"),
            None => "null: no TIMESTAMP_QUERY on this device".into(),
        };
        writeln!(
            f,
            "{:<10} {:>7} {:>9} {:>9} {:>9} {:>9}  source",
            "pass", "frames", "mean ms", "p50 ms", "p95 ms", "max ms"
        )?;
        for (p, name) in PASSES.iter().enumerate() {
            let v: Vec<f64> = self.gpu.iter().filter_map(|t| t[p]).collect();
            match stats(v.clone()) {
                Some((mean, p50, p95, max)) => writeln!(
                    f,
                    "{name:<10} {:>7} {mean:>9.3} {p50:>9.3} {p95:>9.3} {max:>9.3}  GPU, {source}",
                    v.len()
                )?,
                None => writeln!(
                    f,
                    "{name:<10} {:>7} {:>9} {:>9} {:>9} {:>9}  GPU, {source} (or pass not run)",
                    0, "null", "null", "null", "null"
                )?,
            }
        }
        if let Some((mean, p50, p95, max)) = stats(self.cpu_ms.clone()) {
            writeln!(
                f,
                "{:<10} {:>7} {mean:>9.3} {p50:>9.3} {p95:>9.3} {max:>9.3}  CPU record+submit, std::time::Instant",
                "cpu",
                self.cpu_ms.len()
            )?;
        }
        writeln!(
            f,
            "set_scene {:.1} ms · first frame (pipelines) {:.1} ms · Instant",
            self.set_scene_ms, self.first_frame_ms
        )?;
        match &self.pipeline_cache {
            Some((file, n)) => writeln!(f, "pipeline cache: {n} bytes saved to {file}")?,
            None => writeln!(f, "pipeline cache: off (no PIPELINE_CACHE, or X3D_PIPELINE_CACHE=off)")?,
        }
        writeln!(
            f,
            "draws: {} in scene · {} issued · {} CPU-culled · GPU kept {} · {} shadow passes (last frame)",
            self.draws_total,
            self.draw_calls,
            self.culled,
            self.gpu_visible.map_or("n/a".into(), |n| n.to_string()),
            self.shadow_passes
        )
    }
}
