//! x3d.wgpu in the browser.
//!
//! The renderer is the same `x3d-render` crate the desktop viewer uses,
//! compiled to WebAssembly. This crate only supplies what a browser needs:
//! a canvas-backed surface, an `async` start-up, and a small JS-facing API
//! (`Viewer`). The page's JavaScript owns the animation loop and the input
//! events and calls [`Viewer::frame`] once per `requestAnimationFrame`.
//!
//! **Backends.** `"webgpu"` asks the browser's WebGPU for an adapter and
//! fails *before touching the canvas* when there is none (WebGL cannot be probed that way), so the page can
//! retry on the same canvas with `"gl"` (wgpu's WebGL2 backend, with the
//! `Profile::Weak` budget: no HDR, no compute culling). A canvas can only
//! ever hold one kind of context, which is why the probe comes first.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use x3d_render::headless::camera_for;
use x3d_render::{Gpu, OrbitCamera, Profile, RenderSettings, Renderer, ToneMap};
use x3d_scene::Scene;

fn js_err(msg: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&msg.to_string())
}

/// One canvas, one GPU, one scene.
#[wasm_bindgen]
pub struct Viewer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    scene: Scene,
    camera: OrbitCamera,
    settings: RenderSettings,
    backend: &'static str,
    msaa: u32,
}

#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&JsValue::from_str(&info.to_string()))));
}

#[wasm_bindgen]
impl Viewer {
    /// `backend`: `"webgpu"` or `"gl"`. Rejects (without using the canvas) when
    /// the browser has no adapter for it.
    pub async fn create(canvas: web_sys::HtmlCanvasElement, backend: String) -> Result<Viewer, JsValue> {
        let (backends, profile, name): (wgpu::Backends, Profile, &'static str) = match backend.as_str() {
            "webgpu" => (wgpu::Backends::BROWSER_WEBGPU, Profile::Auto, "WebGPU"),
            "gl" => (wgpu::Backends::GL, Profile::Weak, "WebGL2 (wgpu GL backend)"),
            other => return Err(js_err(format!("unknown backend {other:?}, use \"webgpu\" or \"gl\""))),
        };
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        // WebGPU: probe first, so "no adapter" leaves the canvas untouched. (WebGL has
        // no adapter without a canvas: its GL context is created from the surface.)
        if backend == "webgpu" {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .map_err(|e| js_err(format!("no {name} adapter: {e}")))?;
        }

        let surface = instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone())).map_err(js_err)?;
        let gpu = Gpu::new(instance, x3d_render::Backend::Auto, profile, Some(&surface)).await.map_err(js_err)?;

        let caps = surface.get_capabilities(&gpu.adapter);
        let first = *caps.formats.first().ok_or_else(|| js_err("the canvas offers no surface formats"))?;
        // Our shader writes display-ready values: prefer a non-sRGB format.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(first);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: Default::default(),
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);
        let msaa = 4;
        let renderer = Renderer::new(&gpu, format, msaa);
        let scene = x3d_scene::parse_str("<X3D version='4.0'><Scene/></X3D>").map_err(js_err)?;
        let camera = camera_for(&scene, None);
        Ok(Viewer {
            surface,
            config,
            gpu,
            renderer,
            scene,
            camera,
            settings: RenderSettings::default(),
            backend: name,
            msaa,
        })
    }

    /// Replace the scene with X3D (XML encoding) text. Returns a one-line
    /// summary; a parse error comes back as the rejection, with its line.
    pub fn load_scene(&mut self, xml: &str) -> Result<String, JsValue> {
        let mut scene = x3d_scene::parse_str(xml).map_err(js_err)?;
        scene.advance(0.0);
        self.renderer.set_scene(&scene);
        self.camera = camera_for(&scene, None);
        let st = scene.report.stats;
        let line = format!(
            "{} tris, {} line segments, {} ROUTEs",
            st.triangles,
            st.line_segments,
            scene.animation.routes.len()
        );
        self.scene = scene;
        Ok(line)
    }

    /// Draw one frame; `seconds` is the animation clock.
    pub fn frame(&mut self, seconds: f64) -> Result<(), JsValue> {
        if !self.scene.animation.is_empty() {
            let changes = self.scene.advance(seconds);
            self.renderer.update(&self.scene, &changes);
        }
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        self.renderer.render(&view, (self.config.width, self.config.height), &self.camera, &self.settings);
        self.gpu.queue.present(frame);
        Ok(())
    }

    /// Call when the canvas's pixel size changed.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.gpu.device, &self.config);
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy, self.config.height as f32);
    }
    pub fn zoom(&mut self, steps: f32) {
        self.camera.zoom(steps);
    }

    /// `shadows`, `bloom`, `exposure`, `tone_map` (0 off, 1 ACES, 2 AgX),
    /// `cull_back_faces`. The GPU budget may still veto a setting (the page
    /// can read [`Viewer::budget`]).
    pub fn set_option(&mut self, key: &str, value: f64) -> Result<(), JsValue> {
        match key {
            "shadows" => self.settings.shadows = value != 0.0,
            "bloom" => self.settings.bloom = value as f32,
            "exposure" => self.settings.exposure = value as f32,
            "cull_back_faces" => self.settings.cull_back_faces = value != 0.0,
            "tone_map" => {
                self.settings.tone_map = match value as i32 {
                    1 => ToneMap::Aces,
                    2 => ToneMap::Agx,
                    _ => ToneMap::Off,
                }
            }
            other => return Err(js_err(format!("unknown option {other:?}"))),
        }
        Ok(())
    }

    /// Which API actually runs, and which adapter, from `adapter.get_info()`.
    pub fn adapter(&self) -> String {
        let i = self.gpu.adapter.get_info();
        format!("{} on {} ({:?}, driver {:?})", self.backend, i.name, i.device_type, i.driver)
    }

    /// What the budget switched on or off, one line.
    pub fn budget(&self) -> String {
        let b = self.renderer.budget();
        format!(
            "tier {:?}; sun shadow {}² x{} cascades; local-light shadow layers {}; HDR/tone map/bloom {}; GPU culling {}; max MSAA {}x (decisions and their sources: Budget in the console)",
            b.tier,
            b.sun_shadow_size,
            b.cascades,
            b.local_shadow_layers,
            if b.hdr { "on" } else { "off" },
            if b.gpu_cull { "on" } else { "off" },
            b.max_msaa
        )
    }

    pub fn sample_count(&self) -> u32 {
        self.renderer.sample_count().min(self.msaa)
    }
}

impl std::fmt::Debug for Viewer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Viewer")
            .field("backend", &self.backend)
            .field("size", &(self.config.width, self.config.height))
            .finish()
    }
}
