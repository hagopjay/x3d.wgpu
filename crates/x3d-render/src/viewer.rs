//! The interactive window.
//!
//! 🦀 **winit's model.** You hand winit an object that implements
//! [`ApplicationHandler`]; it calls your methods as events arrive
//! (`resumed`, `window_event`, `user_event`…). Our [`App`] keeps an
//! `Option<State>`: `None` until the OS gives us a window.
//!
//! 🦀 **Threads.** A small watcher thread polls file modification times and
//! wakes the event loop through an [`EventLoopProxy`] when the scene or the
//! shader changes on disk. The list of watched files lives in an
//! `Arc<Mutex<…>>`, shared between the two threads.
//!
//! Controls are listed in [`HELP`].

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};
use x3d_scene::{LoadOptions, Scene};

use crate::budget::Profile;
use crate::camera::OrbitCamera;
use crate::gpu::{Backend, Gpu};
use crate::headless::{Offscreen, camera_for};
use crate::renderer::{ColorMode, RenderSettings, Renderer, ToneMap};

pub const HELP: &str = "\
mouse  left-drag orbit · right/middle-drag pan · wheel zoom · drop a .x3d file to open it
1-7    view: lit · normals · depth · base colour · lighting only · facing (green front / red back) · uv
Tab    next Viewpoint      F  frame everything    R  reset to the first Viewpoint
W      wireframe           H  headlight on/off    C  colour mode (display ↔ linear)
B      back-face culling   L  lines               P  points
+ / -  exposure             [ / ]  thinner/thicker lines and points    Space  pause/run animation
T      tone map (off · ACES · AgX)   G  bloom (glow)   O  sun shadows
S      save screenshot     F5 reload              Esc quit";

#[derive(Debug, Clone)]
pub struct ViewerOptions {
    pub scene: PathBuf,
    /// Load the WGSL from this file (and hot-reload it) instead of the built-in copy.
    pub shader: Option<PathBuf>,
    pub backend: Backend,
    pub profile: Profile,
    pub msaa: u32,
    pub size: (u32, u32),
    pub settings: RenderSettings,
    pub load: LoadOptions,
    /// For CI / screenshots: save a PNG here after `exit_after` frames and quit.
    pub screenshot: Option<PathBuf>,
    pub exit_after: Option<u32>,
}

#[derive(Debug)]
enum UserEvent {
    FileChanged(PathBuf),
}

/// Open the window and run until it closes.
pub fn run(opts: ViewerOptions) -> Result<()> {
    let scene =
        x3d_scene::load_with(&opts.scene, &opts.load).with_context(|| format!("loading {}", opts.scene.display()))?;
    print_report(&opts.scene, &scene);
    println!("{HELP}");

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    // Wait for events instead of spinning: the GPU only works when something changes.
    event_loop.set_control_flow(ControlFlow::Wait);

    let watched =
        Arc::new(Mutex::new(std::iter::once(opts.scene.clone()).chain(opts.shader.clone()).collect::<Vec<_>>()));
    spawn_watcher(Arc::clone(&watched), event_loop.create_proxy());

    let mut app = App { opts, scene, watched, state: None, failure: None };
    event_loop.run_app(&mut app)?;
    // 🦀 Errors inside callbacks can't be returned through winit, so we
    // park the first one and return it here.
    app.failure.map_or(Ok(()), Err)
}

fn print_report(path: &Path, scene: &Scene) {
    println!("── {} ──\n{}", path.display(), scene.report);
}

/// Poll modification times; tell the event loop when a file changes.
fn spawn_watcher(paths: Arc<Mutex<Vec<PathBuf>>>, proxy: EventLoopProxy<UserEvent>) {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    std::thread::Builder::new()
        .name("file watcher".into())
        .spawn(move || {
            let mut last: Vec<(PathBuf, Option<SystemTime>)> = Vec::new();
            loop {
                std::thread::sleep(Duration::from_millis(250));
                // 🦀 Hold the lock only long enough to copy the list.
                let now: Vec<PathBuf> = paths.lock().expect("watch list lock").clone();
                for p in now {
                    let t = mtime(&p);
                    match last.iter_mut().find(|(q, _)| *q == p) {
                        Some((_, old)) if *old != t => {
                            *old = t;
                            // An error means the event loop is gone: stop watching.
                            if proxy.send_event(UserEvent::FileChanged(p.clone())).is_err() {
                                return;
                            }
                        }
                        Some(_) => {}
                        None => last.push((p, t)),
                    }
                }
            }
        })
        .expect("spawn watcher thread");
}

struct App {
    opts: ViewerOptions,
    scene: Scene,
    watched: Arc<Mutex<Vec<PathBuf>>>,
    state: Option<State>,
    failure: Option<anyhow::Error>,
}

/// Everything that exists only while we have a window.
struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    camera: OrbitCamera,
    settings: RenderSettings,
    viewpoint: usize,
    input: Input,
    frames: u32,
    title_updated: Instant,
    last_cpu_ms: f32,
    shader_error: bool,
    /// Animation clock: seconds of scene time, and whether it's running.
    clock: Clock,
}

/// 🦀 A tiny state machine: running from an instant, or paused at a time.
#[derive(Debug, Clone, Copy)]
enum Clock {
    Running { since: Instant, offset: f64 },
    Paused { at: f64 },
}

impl Clock {
    fn now(self) -> f64 {
        match self {
            Self::Running { since, offset } => offset + since.elapsed().as_secs_f64(),
            Self::Paused { at } => at,
        }
    }
    fn toggle(self) -> Self {
        match self {
            Self::Running { .. } => Self::Paused { at: self.now() },
            Self::Paused { at } => Self::Running { since: Instant::now(), offset: at },
        }
    }
}

#[derive(Default)]
struct Input {
    cursor: Option<(f64, f64)>,
    orbiting: bool,
    panning: bool,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<State> {
        let attrs = Window::default_attributes()
            .with_title("x3d-view")
            .with_inner_size(PhysicalSize::new(self.opts.size.0, self.opts.size.1));
        let window = Arc::new(event_loop.create_window(attrs)?);

        // 🦀 The surface borrows the window. Giving it an `Arc<Window>` (an
        // owned handle) makes it `Surface<'static>`: no lifetime to juggle.
        let instance = crate::gpu::make_instance(self.opts.backend);
        let surface = instance.create_surface(Arc::clone(&window))?;
        let gpu = pollster::block_on(Gpu::new(instance, self.opts.backend, self.opts.profile, Some(&surface)))?;
        log::info!("GPU: {}", gpu.describe());
        println!("GPU: {}", gpu.describe());

        let caps = surface.get_capabilities(&gpu.adapter);
        // Prefer a non-sRGB format: our shader writes display-ready values.
        // (Renderer handles sRGB formats too, by decoding before the write.)
        let Some(&first) = caps.formats.first() else {
            anyhow::bail!("this surface offers no formats on {} (is the window on another GPU?)", gpu.describe());
        };
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(first);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: Default::default(),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let mut renderer = Renderer::new(&gpu, format, self.opts.msaa);
        let mut shader_error = false;
        if let Some(path) = &self.opts.shader {
            shader_error = load_shader(&mut renderer, path).is_err();
        }
        self.scene.advance(0.0);
        renderer.set_scene(&self.scene);
        let camera = camera_for(&self.scene, None);
        Ok(State {
            window,
            surface,
            config,
            gpu,
            renderer,
            camera,
            settings: self.opts.settings,
            viewpoint: 0,
            input: Input::default(),
            frames: 0,
            title_updated: Instant::now() - Duration::from_secs(1),
            last_cpu_ms: 0.0,
            shader_error,
            clock: Clock::Running { since: Instant::now(), offset: 0.0 },
        })
    }

    fn reload_scene(&mut self, path: &Path) {
        match x3d_scene::load_with(path, &self.opts.load) {
            Ok(mut scene) => {
                print_report(path, &scene);
                if let Some(s) = &mut self.state {
                    scene.advance(s.clock.now());
                    s.renderer.set_scene(&scene);
                    if path != self.opts.scene {
                        // A different file: start from its own viewpoint.
                        s.camera = camera_for(&scene, None);
                        s.viewpoint = 0;
                    }
                    s.window.request_redraw();
                }
                self.scene = scene;
                if path != self.opts.scene {
                    let mut w = self.watched.lock().expect("watch list lock");
                    w.retain(|p| *p != self.opts.scene);
                    w.push(path.to_owned());
                    self.opts.scene = path.to_owned();
                }
            }
            // Keep showing the old scene; just say what's wrong.
            Err(e) => eprintln!("⚠ {}: {e:#}", path.display()),
        }
    }

    fn key(&mut self, key: &Key, event_loop: &ActiveEventLoop) {
        let Some(s) = &mut self.state else { return };
        let views = crate::renderer::DebugView::ALL;
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Tab) => {
                if !self.scene.viewpoints.is_empty() {
                    s.viewpoint = (s.viewpoint + 1) % self.scene.viewpoints.len();
                    s.camera = camera_for(&self.scene, Some(s.viewpoint));
                    // Like a browser: binding a viewpoint fires its bindTime events.
                    self.scene.bind_viewpoint(s.viewpoint);
                    let vp = &self.scene.viewpoints[s.viewpoint];
                    println!("viewpoint {}: {}", s.viewpoint, vp.description);
                }
            }
            Key::Named(NamedKey::Space) => {
                s.clock = s.clock.toggle();
                println!("animation {}", if matches!(s.clock, Clock::Paused { .. }) { "paused" } else { "running" });
            }
            Key::Named(NamedKey::F5) => {
                let path = self.opts.scene.clone();
                self.reload_scene(&path);
                return;
            }
            Key::Character(c) => match c.to_lowercase().as_str() {
                d @ ("1" | "2" | "3" | "4" | "5" | "6" | "7") => {
                    let i: usize = d.parse().expect("a digit");
                    s.settings.view = views[i - 1];
                }
                "v" => s.settings.view = s.settings.view.next(),
                "w" => {
                    if s.renderer.supports_wireframe() {
                        s.settings.wireframe = !s.settings.wireframe;
                    } else {
                        println!("this GPU has no POLYGON_MODE_LINE: wireframe unavailable");
                    }
                }
                "h" => {
                    let on = s.settings.headlight.unwrap_or(self.scene.headlight);
                    s.settings.headlight = Some(!on);
                }
                "]" => s.settings.line_scale *= 1.25,
                "[" => s.settings.line_scale = (s.settings.line_scale / 1.25).max(0.25),
                "+" | "=" => s.settings.exposure *= 1.25,
                "-" => s.settings.exposure /= 1.25,
                "c" => {
                    s.settings.color_mode = match s.settings.color_mode {
                        ColorMode::Display => ColorMode::Linear,
                        ColorMode::Linear => ColorMode::Display,
                    };
                }
                "b" => s.settings.cull_back_faces = !s.settings.cull_back_faces,
                "t" => s.settings.tone_map = s.settings.tone_map.next(),
                "g" => s.settings.bloom = if s.settings.bloom > 0.0 { 0.0 } else { 0.15 },
                "o" => s.settings.shadows = !s.settings.shadows,
                "l" => s.settings.show_lines = !s.settings.show_lines,
                "p" => s.settings.show_points = !s.settings.show_points,
                "f" => s.camera = OrbitCamera::framing(self.scene.bounds()),
                "r" => {
                    s.viewpoint = 0;
                    s.camera = camera_for(&self.scene, None);
                }
                "s" => {
                    let name = format!("screenshot-{}.png", chrono_free_stamp());
                    match screenshot(s, Path::new(&name)) {
                        Ok(()) => println!("saved {name}"),
                        Err(e) => eprintln!("screenshot failed: {e:#}"),
                    }
                }
                _ => return,
            },
            _ => return,
        }
        s.title_updated = Instant::now() - Duration::from_secs(1); // refresh the title now
        s.window.request_redraw();
    }
}

/// Seconds since 1970: unique enough for screenshot names, no extra crate.
fn chrono_free_stamp() -> u64 {
    SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn load_shader(renderer: &mut Renderer, path: &Path) -> Result<(), ()> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("⚠ cannot read {}: {e}", path.display());
            return Err(());
        }
    };
    match renderer.reload_shader(&source) {
        Ok(()) => {
            println!("✔ shader {} loaded", path.display());
            Ok(())
        }
        Err(e) => {
            eprintln!("⚠ shader {} rejected; keeping the previous one.\n{e}", path.display());
            Err(())
        }
    }
}

/// Render the current view off-screen at window size and save it.
fn screenshot(s: &mut State, path: &Path) -> Result<()> {
    let size = (s.config.width, s.config.height);
    let target = Offscreen::new(&s.gpu.device, size, s.config.format);
    s.renderer.render(&target.view, size, &s.camera, &s.settings);
    target.read(&s.gpu)?.save(path)?;
    Ok(())
}

impl State {
    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return; // minimised
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.window.request_redraw();
    }

    fn draw(&mut self, scene_name: &str, stats_line: &str) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                // The window changed under us: reconfigure and try next frame.
                self.surface.configure(&self.gpu.device, &self.config);
                self.window.request_redraw();
                return;
            }
            // Timeout / Occluded: skip this frame; the next redraw retries.
            other => {
                log::warn!("no frame this time: {other:?}");
                self.window.request_redraw();
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let size = (self.config.width, self.config.height);
        let stats = self.renderer.render(&view, size, &self.camera, &self.settings);
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        self.frames += 1;
        self.last_cpu_ms = stats.cpu_ms;

        if self.title_updated.elapsed() > Duration::from_millis(500) {
            self.title_updated = Instant::now();
            let s = &self.settings;
            let flags = [
                (s.wireframe, "wire"),
                (s.color_mode == ColorMode::Linear, "linear"),
                (!s.cull_back_faces, "no-cull"),
                (s.tone_map == ToneMap::Aces, "ACES"),
                (s.tone_map == ToneMap::Agx, "AgX"),
                (s.bloom > 0.0, "bloom"),
                (s.shadows, "shadows"),
                (self.shader_error, "⚠shader"),
            ];
            let flags: Vec<&str> = flags.iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect();
            self.window.set_title(&format!(
                "x3d-view — {scene_name} — {} — {stats_line} — {} draws, {} lights, {:.2} ms CPU {}",
                s.view,
                stats.draw_calls,
                stats.lights,
                self.last_cpu_ms,
                flags.join(" ")
            ));
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match self.init(event_loop) {
            Ok(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
            Err(e) => {
                self.failure = Some(e);
                event_loop.exit();
            }
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::FileChanged(path) = event;
        if self.opts.shader.as_deref() == Some(path.as_path()) {
            if let Some(s) = &mut self.state {
                s.shader_error = load_shader(&mut s.renderer, &path).is_err();
                s.window.request_redraw();
            }
        } else {
            println!("↻ {} changed", path.display());
            self.reload_scene(&path);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(s) = &mut self.state {
                    s.resize(size);
                }
            }
            WindowEvent::DroppedFile(path) => self.reload_scene(&path),
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !event.repeat => {
                self.key(&event.logical_key, event_loop);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(s) = &mut self.state {
                    let down = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => s.input.orbiting = down,
                        MouseButton::Right | MouseButton::Middle => s.input.panning = down,
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(s) = &mut self.state {
                    let p = (position.x, position.y);
                    if let Some((x, y)) = s.input.cursor {
                        let (dx, dy) = ((p.0 - x) as f32, (p.1 - y) as f32);
                        if s.input.orbiting {
                            s.camera.orbit(dx, dy);
                            s.window.request_redraw();
                        } else if s.input.panning {
                            s.camera.pan(dx, dy, s.config.height as f32);
                            s.window.request_redraw();
                        }
                    }
                    s.input.cursor = Some(p);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(s) = &mut self.state {
                    let steps = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                    };
                    s.camera.zoom(steps);
                    s.window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // 🎨 Device lost (driver reset, GPU removed, TDR): everything
                // made from the old device is dead. Drop it all and build a
                // new window state on a fresh adapter + device.
                if self.state.as_ref().is_some_and(|s| s.gpu.is_lost()) {
                    log::error!("GPU device lost: rebuilding the renderer");
                    self.state = None;
                    let started = std::time::Instant::now();
                    match self.init(event_loop) {
                        Ok(state) => {
                            log::info!("recovered from device loss in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
                            state.window.request_redraw();
                            self.state = Some(state);
                        }
                        Err(e) => {
                            self.failure = Some(e.context("recovering from a lost GPU device"));
                            event_loop.exit();
                        }
                    }
                    return;
                }
                let name = self.opts.scene.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let Some(s) = &mut self.state else { return };
                // Animation: run the event graph to "now", push what changed.
                let animated = !self.scene.animation.is_empty();
                let t = s.clock.now();
                if animated {
                    let changes = self.scene.advance(t);
                    s.renderer.update(&self.scene, &changes);
                }
                let st = self.scene.report.stats;
                let mut stats_line = format!("{} tris, {} lines", st.triangles, st.line_segments);
                if animated {
                    stats_line +=
                        &format!(", t {t:.1} s{}", if matches!(s.clock, Clock::Paused { .. }) { " ⏸" } else { "" });
                }
                s.draw(&name, &stats_line);
                // Keep frames coming while something moves (Wait would stall otherwise).
                if animated && matches!(s.clock, Clock::Running { .. }) {
                    s.window.request_redraw();
                }

                // Scripted runs (CI, Xvfb): screenshot after N frames, then quit.
                if let Some(n) = self.opts.exit_after {
                    if s.frames >= n {
                        if let Some(path) = &self.opts.screenshot {
                            match screenshot(s, path) {
                                Ok(()) => println!("saved {}", path.display()),
                                Err(e) => self.failure = Some(e),
                            }
                        }
                        event_loop.exit();
                    } else {
                        s.window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}
