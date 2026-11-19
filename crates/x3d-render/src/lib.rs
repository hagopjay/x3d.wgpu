//! # x3d-render
//!
//! A small, readable wgpu renderer for [`x3d_scene`] scenes. On Linux it
//! runs on **Vulkan** (pass `--backend vulkan` to insist); wgpu also speaks
//! Metal, DX12 and GL, and every line here is safe Rust.
//!
//! ```text
//!  .x3d ──x3d_scene::load──▶ Scene ──Renderer::set_scene──▶ GPU buffers
//!                                         │
//!            OrbitCamera + RenderSettings ┴─Renderer::render──▶ texture
//!                                                     │
//!                                   window (viewer) ◀─┴─▶ PNG (headless)
//! ```
//!
//! Start with `examples/hello_triangle.rs`, then read the modules in this
//! order: [`gpu`] → [`uniforms`] → `shaders/scene.wgsl` → [`renderer`] →
//! [`camera`] → [`headless`] → `viewer`.

pub mod bench;
pub mod budget;
pub mod camera;
pub mod conform;
pub mod cull;
pub mod custom;
pub mod diff;
pub mod glsl;
pub mod gpu;
pub mod headless;
pub mod pipeline_cache;
pub mod post;
pub mod renderer;
pub mod shadows;
pub mod textures;
pub mod timing;
pub mod uniforms;
// 🦀 Conditional compilation: this module only exists with the `window` feature.
#[cfg(feature = "window")]
pub mod viewer;

pub use budget::{Budget, Profile};
pub use camera::OrbitCamera;
pub use gpu::{Backend, Gpu};
pub use headless::{Shot, render_scene, render_sheet};
pub use renderer::{ColorMode, DebugView, RenderSettings, Renderer, ToneMap};
