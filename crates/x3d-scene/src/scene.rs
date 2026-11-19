//! The *output* of loading: a flat, GPU-friendly description of a scene.
//!
//! X3D is a tree (Transforms inside Groups inside Switches…). Renderers
//! prefer flat lists. Loading walks the tree once and produces:
//!
//! - [`Mesh`]es — unique vertex/index data. A geometry that is `DEF`'d once
//!   and `USE`'d ten times becomes **one** mesh (shared through `Arc`).
//! - [`Draw`]s — "draw mesh *m* with this world matrix and this material".
//! - [`Light`]s, [`Viewpoint`]s and the [`Background`], already in world space.

use std::path::PathBuf;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3, Vec4};

use crate::report::Report;

/// One vertex as the GPU sees it.
///
/// 🦀 `#[repr(C)]` fixes the field order and padding so the bytes match the
/// vertex layout declared in the renderer. With the `bytemuck` feature the
/// type is also `Pod` ("plain old data"), so a `&[Vertex]` can be viewed as
/// `&[u8]` for free and copied straight into a GPU buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "bytemuck", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// RGBA from a Color/ColorRGBA node, or white.
    pub color: [f32; 4],
    /// Texture coordinate: (s, t, 0) from a TextureCoordinate, or the
    /// object-space position for `TextureCoordinateGenerator mode='COORD'`.
    /// The material's texture matrix turns it into the final (s, t).
    pub texcoord: [f32; 3],
}

// 🦀 A compile-time check: if someone adds a field, this stops the build
// until the GPU vertex layout is updated too.
const _: () = assert!(std::mem::size_of::<Vertex>() == 52);

/// What kind of primitives a mesh's indices describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Topology {
    Triangles,
    Lines,
    Points,
}

/// An axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// The "nothing" box: `min = +∞`, `max = −∞`, so any union replaces it.
    pub const EMPTY: Self = Self {
        min: Vec3::INFINITY,
        max: Vec3::NEG_INFINITY,
    };

    pub fn is_empty(&self) -> bool {
        self.min.cmpgt(self.max).any()
    }
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
    /// The box around this box after a transform (all 8 corners).
    pub fn transformed(&self, m: &Mat4) -> Self {
        if self.is_empty() {
            return *self;
        }
        let (lo, hi) = (self.min, self.max);
        (0..8)
            .map(|i| {
                let corner = Vec3::new(
                    if i & 1 == 0 { lo.x } else { hi.x },
                    if i & 2 == 0 { lo.y } else { hi.y },
                    if i & 4 == 0 { lo.z } else { hi.z },
                );
                m.transform_point3(corner)
            })
            .collect()
    }
}

/// 🦀 Implementing `FromIterator` lets you write `points.iter().collect::<Aabb>()`.
impl FromIterator<Vec3> for Aabb {
    fn from_iter<I: IntoIterator<Item = Vec3>>(iter: I) -> Self {
        iter.into_iter().fold(Self::EMPTY, |b, p| Self {
            min: b.min.min(p),
            max: b.max.max(p),
        })
    }
}

/// CPU-side geometry, ready to upload.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub topology: Topology,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// `solid='true'`: back faces may be culled.
    pub solid: bool,
    /// The geometry carried its own colours (Color/ColorRGBA).
    pub has_colors: bool,
    pub bounds: Aabb,
    /// For logs: the X3D node name and DEF, e.g. `IndexedFaceSet "Hull"`.
    pub label: String,
}

impl Mesh {
    pub fn new(topology: Topology, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let bounds = vertices.iter().map(|v| Vec3::from(v.position)).collect();
        Self {
            topology,
            vertices,
            indices,
            solid: true,
            has_colors: false,
            bounds,
            label: String::new(),
        }
    }

    /// Number of primitives (triangles, line segments or points).
    pub fn primitive_count(&self) -> usize {
        match self.topology {
            Topology::Triangles => self.indices.len() / 3,
            Topology::Lines => self.indices.len() / 2,
            Topology::Points => self.indices.len(),
        }
    }

    /// Merge identical vertices (same position, normal and colour).
    ///
    /// 🦀 `f32` is not `Hash`/`Eq` (NaN ≠ NaN), so we hash the *bit
    /// patterns* instead. `to_bits` turns each float into a `u32`.
    pub fn weld(&mut self) {
        use std::collections::HashMap;
        let key = |v: &Vertex| -> [u32; 13] {
            let mut k = [0u32; 13];
            let all = v
                .position
                .iter()
                .chain(&v.normal)
                .chain(&v.color)
                .chain(&v.texcoord);
            for (slot, f) in k.iter_mut().zip(all) {
                *slot = f.to_bits();
            }
            k
        };
        let mut seen: HashMap<[u32; 13], u32> = HashMap::with_capacity(self.vertices.len());
        let mut unique = Vec::with_capacity(self.vertices.len());
        let remap: Vec<u32> = self
            .vertices
            .iter()
            .map(|v| {
                // 🦀 The entry API: look up once, insert if missing.
                *seen.entry(key(v)).or_insert_with(|| {
                    unique.push(*v);
                    (unique.len() - 1) as u32
                })
            })
            .collect();
        for i in &mut self.indices {
            *i = remap[*i as usize];
        }
        self.vertices = unique;
    }
}

/// An image file used as a texture, already resolved to a path on disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TextureRef {
    pub path: PathBuf,
    pub repeat_s: bool,
    pub repeat_t: bool,
}

/// The texture slots of X3D 4 materials. `None` = no texture in that slot.
///
/// | slot | Material | PhysicalMaterial | UnlitMaterial |
/// |---|---|---|---|
/// | `base` | diffuseTexture, or Appearance.texture | baseTexture | emissiveTexture, or Appearance.texture |
/// | `normal` | normalTexture | normalTexture | – |
/// | `metallic_roughness` | – | metallicRoughnessTexture (G = roughness, B = metallic) | – |
/// | `occlusion` | occlusionTexture | occlusionTexture (R) | – |
/// | `emissive` | emissiveTexture | emissiveTexture | – |
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Textures {
    pub base: Option<TextureRef>,
    pub normal: Option<TextureRef>,
    pub metallic_roughness: Option<TextureRef>,
    pub occlusion: Option<TextureRef>,
    pub emissive: Option<TextureRef>,
}

impl Textures {
    pub const NONE: Self = Self {
        base: None,
        normal: None,
        metallic_roughness: None,
        occlusion: None,
        emissive: None,
    };

    /// The slots in a fixed order (matches the renderer's bind group).
    pub fn slots(&self) -> [Option<&TextureRef>; 5] {
        [
            &self.base,
            &self.normal,
            &self.metallic_roughness,
            &self.occlusion,
            &self.emissive,
        ]
        .map(Option::as_ref)
    }
    pub fn is_empty(&self) -> bool {
        self.slots().iter().all(Option::is_none)
    }
}

/// PhysicalMaterial's metallic-roughness parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pbr {
    pub metallic: f32,
    pub roughness: f32,
    pub occlusion_strength: f32,
    pub normal_scale: f32,
}

/// How a surface is lit, and with what.
///
/// Classic `Material` uses X3D's Blinn-Phong equation; `PhysicalMaterial`
/// (`pbr` is `Some`) uses metallic-roughness PBR; unlit surfaces just show
/// their colour.
#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub diffuse: Vec3,
    pub emissive: Vec3,
    pub specular: Vec3,
    /// 0..1; the exponent is `shininess × 128`.
    pub shininess: f32,
    pub ambient_intensity: f32,
    /// 0 = opaque, 1 = invisible.
    pub transparency: f32,
    /// Unlit: colour = emissive (or the geometry's colours). Lines, points,
    /// UnlitMaterial and shapes without any material are unlit in X3D.
    pub unlit: bool,
    /// `Some` for PhysicalMaterial. `diffuse` then holds `baseColor`.
    pub pbr: Option<Pbr>,
    pub textures: Textures,
    /// TextureTransform / TextureTransformMatrix3D, applied to `Vertex::texcoord`.
    pub texture_matrix: Mat4,
}

impl Default for Material {
    /// The defaults from the X3D spec (Material node).
    fn default() -> Self {
        Self {
            diffuse: Vec3::splat(0.8),
            emissive: Vec3::ZERO,
            specular: Vec3::ZERO,
            shininess: 0.2,
            ambient_intensity: 0.2,
            transparency: 0.0,
            unlit: false,
            pbr: None,
            textures: Textures::NONE,
            texture_matrix: Mat4::IDENTITY,
        }
    }
}

impl Material {
    /// What X3D draws when a Shape has no Material at all: unlit white.
    pub const UNLIT_WHITE: Self = Self {
        diffuse: Vec3::ONE,
        emissive: Vec3::ONE,
        specular: Vec3::ZERO,
        shininess: 0.2,
        ambient_intensity: 0.0,
        transparency: 0.0,
        unlit: true,
        pbr: None,
        textures: Textures::NONE,
        texture_matrix: Mat4::IDENTITY,
    };

    pub fn is_transparent(&self) -> bool {
        self.transparency > 0.0
    }
}

/// One "draw this mesh, here, like this".
#[derive(Debug, Clone)]
pub struct Draw {
    pub mesh: Arc<Mesh>,
    /// Local → world.
    pub transform: Mat4,
    pub material: Material,
    /// The light scopes (groups holding `global='false'` lights) this draw is inside.
    pub scopes: Arc<[u32]>,
    /// The LOD levels this draw belongs to (outermost first). Empty for
    /// most draws. The renderer shows the draw only when, for every tag,
    /// the camera's distance picks that level.
    pub lod: Arc<[LodTag]>,
}

/// "This draw is level `level` of LOD number `group`."
#[derive(Debug, Clone, PartialEq)]
pub struct LodTag {
    pub group: u32,
    pub level: u32,
    /// The LOD's `center`, in world space.
    pub center: Vec3,
    /// X3D `range`: level *i* is used while the distance is below `range[i]`
    /// (and at or above `range[i − 1]`); past the last range, the last level.
    pub range: Arc<[f32]>,
}

impl LodTag {
    /// Which level an LOD with this `range` picks at distance `d`.
    ///
    /// ```
    /// use x3d_scene::scene::LodTag;
    /// assert_eq!(LodTag::level_at(&[10.0, 50.0], 5.0), 0);
    /// assert_eq!(LodTag::level_at(&[10.0, 50.0], 20.0), 1);
    /// assert_eq!(LodTag::level_at(&[10.0, 50.0], 900.0), 2);
    /// ```
    pub fn level_at(range: &[f32], d: f32) -> u32 {
        range.iter().take_while(|&&r| d >= r).count() as u32
    }

    /// Is this tag's level the one picked for a camera at `eye`?
    pub fn active(&self, eye: Vec3) -> bool {
        Self::level_at(&self.range, eye.distance(self.center)) == self.level
    }
}

impl Draw {
    pub fn world_bounds(&self) -> Aabb {
        self.mesh.bounds.transformed(&self.transform)
    }
}

/// The three X3D light types, already in world space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightKind {
    Directional {
        direction: Vec3,
    },
    Point {
        location: Vec3,
        attenuation: Vec3,
        radius: f32,
    },
    Spot {
        location: Vec3,
        direction: Vec3,
        attenuation: Vec3,
        radius: f32,
        /// Radians: full intensity inside this half-angle…
        beam_width: f32,
        /// …and nothing outside this one.
        cut_off_angle: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub kind: LightKind,
    pub color: Vec3,
    pub intensity: f32,
    pub ambient_intensity: f32,
    /// `global='false'` lights only light their own group: draws whose
    /// `scopes` contain this id. `None` lights everything.
    pub scope: Option<u32>,
}

impl Light {
    /// Does this light reach a draw inside these scopes?
    pub fn reaches(&self, scopes: &[u32]) -> bool {
        self.scope.is_none_or(|s| scopes.contains(&s))
    }
}

impl Light {
    /// The same light seen through a transform (used when an Inline is placed).
    pub fn transformed(&self, m: &Mat4) -> Self {
        let dir = |d: Vec3| m.transform_vector3(d).normalize_or(d);
        let radius_scale = m.to_scale_rotation_translation().0.max_element();
        let kind = match self.kind {
            LightKind::Directional { direction } => LightKind::Directional {
                direction: dir(direction),
            },
            LightKind::Point {
                location,
                attenuation,
                radius,
            } => LightKind::Point {
                location: m.transform_point3(location),
                attenuation,
                radius: radius * radius_scale,
            },
            LightKind::Spot {
                location,
                direction,
                attenuation,
                radius,
                beam_width,
                cut_off_angle,
            } => LightKind::Spot {
                location: m.transform_point3(location),
                direction: dir(direction),
                attenuation,
                radius: radius * radius_scale,
                beam_width,
                cut_off_angle,
            },
        };
        Self { kind, ..*self }
    }

    /// The NavigationInfo headlight: a white directional light along the
    /// camera's view direction.
    pub fn headlight(view_direction: Vec3) -> Self {
        Self {
            kind: LightKind::Directional {
                direction: view_direction,
            },
            color: Vec3::ONE,
            intensity: 1.0,
            ambient_intensity: 0.0,
            scope: None,
        }
    }
}

/// A camera position from a `<Viewpoint>`, in world space.
#[derive(Debug, Clone, PartialEq)]
pub struct Viewpoint {
    pub description: String,
    pub position: Vec3,
    pub orientation: Quat,
    /// Radians. X3D applies it to the *smaller* screen dimension.
    pub field_of_view: f32,
    pub center_of_rotation: Vec3,
}

impl Default for Viewpoint {
    fn default() -> Self {
        Self {
            description: String::new(),
            position: Vec3::new(0.0, 0.0, 10.0),
            orientation: Quat::IDENTITY,
            field_of_view: std::f32::consts::FRAC_PI_4,
            center_of_rotation: Vec3::ZERO,
        }
    }
}

impl Viewpoint {
    /// X3D cameras look down their local −Z axis.
    pub fn forward(&self) -> Vec3 {
        self.orientation * Vec3::NEG_Z
    }
    pub fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }
}

/// The first `<Background>`: sky colours from the zenith down.
#[derive(Debug, Clone, PartialEq)]
pub struct Background {
    pub sky_colors: Vec<Vec3>,
    pub sky_angles: Vec<f32>,
    pub ground_colors: Vec<Vec3>,
    pub ground_angles: Vec<f32>,
    pub transparency: f32,
}

impl Default for Background {
    fn default() -> Self {
        Self {
            sky_colors: vec![Vec3::ZERO],
            sky_angles: vec![],
            ground_colors: vec![],
            ground_angles: vec![],
            transparency: 0.0,
        }
    }
}

impl Background {
    /// A single clear colour (the zenith). Gradients are a LEARNING.md quest!
    pub fn clear_color(&self) -> Vec4 {
        let c = self.sky_colors.first().copied().unwrap_or(Vec3::ZERO);
        c.extend(1.0 - self.transparency)
    }
}

/// Everything the renderer needs.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub draws: Vec<Draw>,
    pub lights: Vec<Light>,
    pub viewpoints: Vec<Viewpoint>,
    pub background: Background,
    /// NavigationInfo `headlight` (true unless a scene turns it off).
    pub headlight: bool,
    pub report: Report,
    /// How many LOD nodes the draws' `LodTag`s refer to.
    pub lod_groups: u32,
    /// How many light scopes the lights and draws refer to.
    pub light_scopes: u32,
}

impl Scene {
    /// Add an inlined scene's contents, placed by `m`. Its Background and
    /// NavigationInfo don't bind (X3D: only the main scene's do).
    /// `scopes`: the light scopes the Inline itself sits in.
    pub fn splice(&mut self, child: &Scene, m: Mat4, scopes: &[u32], from: &str) {
        let scope_offset = self.light_scopes;
        // LOD groups are numbered per scene: shift the child's past ours, and
        // move their centres along with everything else.
        let offset = self.lod_groups;
        self.draws.extend(child.draws.iter().map(|d| {
            Draw {
                transform: m * d.transform,
                lod: d
                    .lod
                    .iter()
                    .map(|t| LodTag {
                        group: t.group + offset,
                        center: m.transform_point3(t.center),
                        ..t.clone()
                    })
                    .collect(),
                scopes: scopes
                    .iter()
                    .copied()
                    .chain(d.scopes.iter().map(|s| s + scope_offset))
                    .collect(),
                ..d.clone()
            }
        }));
        self.lod_groups += child.lod_groups;
        self.light_scopes += child.light_scopes;
        self.lights.extend(child.lights.iter().map(|l| Light {
            scope: l.scope.map(|s| s + scope_offset),
            ..l.transformed(&m)
        }));
        let (_, rotation, _) = m.to_scale_rotation_translation();
        self.viewpoints
            .extend(child.viewpoints.iter().map(|v| Viewpoint {
                position: m.transform_point3(v.position),
                orientation: rotation * v.orientation,
                center_of_rotation: m.transform_point3(v.center_of_rotation),
                ..v.clone()
            }));
        self.report.absorb(&child.report, from);
    }

    /// The world-space box around every draw.
    pub fn bounds(&self) -> Aabb {
        self.draws
            .iter()
            .map(Draw::world_bounds)
            .fold(Aabb::EMPTY, Aabb::union)
    }

    /// Unique meshes (by pointer), in first-use order.
    pub fn unique_meshes(&self) -> Vec<Arc<Mesh>> {
        let mut out: Vec<Arc<Mesh>> = Vec::new();
        for d in &self.draws {
            if !out.iter().any(|m| Arc::ptr_eq(m, &d.mesh)) {
                out.push(Arc::clone(&d.mesh));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_from_points_and_transform() {
        let b: Aabb = [Vec3::ZERO, Vec3::ONE].into_iter().collect();
        assert_eq!(b.center(), Vec3::splat(0.5));
        let moved = b.transformed(&Mat4::from_translation(Vec3::X * 10.0));
        assert_eq!(moved.min, Vec3::new(10.0, 0.0, 0.0));
        assert!(Aabb::EMPTY.is_empty());
    }

    #[test]
    fn weld_merges_duplicates() {
        let v = Vertex {
            position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            color: [1.0; 4],
            texcoord: [0.0; 3],
        };
        let mut m = Mesh::new(Topology::Triangles, vec![v, v, v], vec![0, 1, 2]);
        m.weld();
        assert_eq!(m.vertices.len(), 1);
        assert_eq!(m.indices, [0, 0, 0]);
    }
}
