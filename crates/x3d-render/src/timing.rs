//! GPU time per pass, from timestamp queries.
//!
//! 🎨 CPU timers only see how long it took to *record* commands; the GPU
//! runs them later, in parallel. To time the GPU you ask it to write its
//! own clock into a *query set* at the start and end of each pass, copy
//! those ticks into a buffer, read them back, and multiply by the queue's
//! timestamp period (nanoseconds per tick). Reading back is asynchronous:
//! the viewer shows the newest finished frame's numbers; `bench` waits.
//!
//! Following the Loom's rule for receipts, a pass that wasn't timed is
//! `None` ("null, never a substitute"), never a CPU number in disguise.

use std::sync::{Arc, Mutex};

/// The passes we time, in order. Shadow and bloom span several passes.
pub const PASSES: [&str; 5] = ["shadow", "cull", "scene", "bloom", "tone map"];
pub const SHADOW: usize = 0;
pub const CULL: usize = 1;
pub const SCENE: usize = 2;
pub const BLOOM: usize = 3;
pub const TONE_MAP: usize = 4;
const QUERIES: u32 = 2 * PASSES.len() as u32;

/// Milliseconds per pass; `None` = didn't run, or not timed.
pub type PassTimes = [Option<f64>; PASSES.len()];

#[derive(Debug)]
enum State {
    Idle,
    Mapping,
    Mapped,
}

#[derive(Debug)]
pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per tick (`Queue::get_timestamp_period`).
    pub period_ns: f32,
    state: Arc<Mutex<State>>,
    /// Which passes wrote timestamps in the frame being read back.
    ran: [bool; PASSES.len()],
    armed: bool,
    pub last: Option<PassTimes>,
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let size = u64::from(QUERIES) * 8;
        Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            period_ns: queue.get_timestamp_period(),
            state: Arc::new(Mutex::new(State::Idle)),
            ran: [false; PASSES.len()],
            armed: false,
            last: None,
        }
    }

    /// Start a frame: timestamps are written only if the last readback is done.
    pub fn begin_frame(&mut self) -> bool {
        self.armed = matches!(*self.state.lock().expect("timer state"), State::Idle);
        if self.armed {
            self.ran = [false; PASSES.len()];
        }
        self.armed
    }

    /// Writes for one pass that is the whole of `pass`.
    pub fn pass(&mut self, pass: usize) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.armed.then(|| {
            self.ran[pass] = true;
            wgpu::RenderPassTimestampWrites {
                query_set: &self.set,
                beginning_of_pass_write_index: Some(2 * pass as u32),
                end_of_pass_write_index: Some(2 * pass as u32 + 1),
            }
        })
    }

    /// For `pass` made of several render passes: (first's begin, last's end).
    pub fn span(
        &mut self,
        pass: usize,
    ) -> (Option<wgpu::RenderPassTimestampWrites<'_>>, Option<wgpu::RenderPassTimestampWrites<'_>>) {
        if !self.armed {
            return (None, None);
        }
        self.ran[pass] = true;
        let at = |b: Option<u32>, e: Option<u32>| wgpu::RenderPassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: b,
            end_of_pass_write_index: e,
        };
        (Some(at(Some(2 * pass as u32), None)), Some(at(None, Some(2 * pass as u32 + 1))))
    }

    pub fn compute(&mut self, pass: usize) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.armed.then(|| {
            self.ran[pass] = true;
            wgpu::ComputePassTimestampWrites {
                query_set: &self.set,
                beginning_of_pass_write_index: Some(2 * pass as u32),
                end_of_pass_write_index: Some(2 * pass as u32 + 1),
            }
        })
    }

    /// After the passes: copy the ticks out (before `finish`).
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.armed && self.ran.iter().any(|&r| r) {
            encoder.resolve_query_set(&self.set, 0..QUERIES, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, u64::from(QUERIES) * 8);
        }
    }

    /// After `submit`: ask for the readback.
    pub fn after_submit(&mut self) {
        if !(self.armed && self.ran.iter().any(|&r| r)) {
            return;
        }
        *self.state.lock().expect("timer state") = State::Mapping;
        let state = self.state.clone();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *state.lock().expect("timer state") = if r.is_ok() { State::Mapped } else { State::Idle };
        });
    }

    /// Pick up finished numbers, if any (call any time; never blocks).
    pub fn collect(&mut self) -> Option<PassTimes> {
        if !matches!(*self.state.lock().expect("timer state"), State::Mapped) {
            return None;
        }
        let ticks: Option<Vec<u64>> = self
            .readback
            .slice(..)
            .get_mapped_range()
            .ok()
            .map(|data| data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect());
        self.readback.unmap();
        *self.state.lock().expect("timer state") = State::Idle;
        let ticks = ticks?;
        let mut out: PassTimes = [None; PASSES.len()];
        for (p, slot) in out.iter_mut().enumerate() {
            let (b, e) = (ticks[2 * p], ticks[2 * p + 1]);
            if self.ran[p] && e >= b && b != 0 {
                *slot = Some((e - b) as f64 * f64::from(self.period_ns) / 1e6);
            }
        }
        self.last = Some(out);
        Some(out)
    }
}
