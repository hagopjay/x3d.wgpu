//! # x3d-scene
//!
//! Load the *static, easily convertible* subset of X3D (XML encoding) into
//! flat, GPU-ready data. No GPU code lives here, so this crate builds and
//! tests anywhere.
//!
//! ```no_run
//! let scene = x3d_scene::load("scenes/still_life.x3d")?;
//! println!("{}", scene.report); // what was kept, approximated or skipped
//! for draw in &scene.draws {
//!     println!("{} → {} triangles", draw.mesh.label, draw.mesh.primitive_count());
//! }
//! # Ok::<(), x3d_scene::Error>(())
//! ```
//!
//! ## What's supported
//! | Area | Nodes |
//! |---|---|
//! | Grouping | Transform, Group, StaticGroup, Switch, Collision, Anchor, CAD*, (Billboard, LOD approximated) |
//! | Geometry | Box, Sphere, Cylinder, Cone, IndexedFaceSet, (Indexed)TriangleSet/FanSet/StripSet, (Indexed)QuadSet, ElevationGrid, IndexedLineSet, LineSet, PointSet |
//! | Appearance | Material, UnlitMaterial, TwoSidedMaterial (front), PhysicalMaterial (approximated) |
//! | Per-vertex | Coordinate, Color, ColorRGBA, Normal, creaseAngle, ccw, solid |
//! | Environment | DirectionalLight, PointLight, SpotLight, Viewpoint, NavigationInfo.headlight, Background (zenith) |
//! | Reuse | DEF / USE (shared geometry is built once) |
//!
//! Everything else is counted in [`Report`], never silently dropped.

// 🦀 Module tree: each `mod` is a file in src/. `pub mod` is part of the API.
pub mod error;
pub mod fields;
pub mod geometry;
mod parse;
pub mod report;
pub mod runtime;
pub mod scene;

// 🦀 Re-exports: users write `x3d_scene::Scene`, not `x3d_scene::scene::Scene`.
pub use error::{Error, Result};
pub use parse::{LoadOptions, load, load_with, parse_str, parse_str_with};
pub use report::{Report, Stats};
pub use scene::{
    Aabb, Background, CubeMap, Draw, Environment, Light, LightKind, LodTag, Material, Mesh, Pbr, Scene, ShaderSource,
    SwitchGuard, TextureRef, Textures, Topology, Vertex, Viewpoint,
};
