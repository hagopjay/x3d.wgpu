//! Turning X3D geometry into triangles, line segments and points.
//!
//! 🎨 **Graphics concepts in this file**
//! - *Fan triangulation*: a convex polygon `0 1 2 3 4` becomes triangles
//!   `(0 1 2) (0 2 3) (0 3 4)`.
//! - *Newell's method* for a polygon's normal: robust even for slightly
//!   non-planar faces.
//! - *Crease angle*: X3D's rule for smooth vs. faceted shading. Two faces
//!   that meet at an angle smaller than `creaseAngle` share a smooth normal.
//! - *Winding order*: front faces are counter-clockwise (CCW) when seen
//!   from the front. The GPU uses this to cull back faces.
//!
//! The intermediate [`Polygons`] type is shared by IndexedFaceSet,
//! the triangle sets and ElevationGrid, so normals and colours are
//! handled in exactly one place.

use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec3, Vec4};

use crate::scene::{Mesh, Topology, Vertex};

/// How an attribute (colour or normal) is attached to a polygon mesh.
///
/// 🦀 An enum *with data*: each variant carries exactly what it needs and
/// the compiler makes every `match` handle all three.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Binding<T> {
    #[default]
    None,
    /// One value per face.
    PerFace(Vec<T>),
    /// One value per corner, laid out exactly like `Polygons::faces`.
    PerCorner(Vec<Vec<T>>),
}

/// Why an index list doesn't fit the data it indexes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what} index {index} is out of range (only {len} values)")]
pub struct IndexError {
    pub what: &'static str,
    pub index: i64,
    pub len: usize,
}

/// Split an X3D index list at `-1`: `0 1 2 -1 2 3 0` → `[[0,1,2],[2,3,0]]`.
/// A missing final `-1` is allowed; empty groups are kept (they still
/// count as faces for per-face colours).
pub fn split_at_minus_one(index: &[i32]) -> Vec<Vec<i32>> {
    if index.is_empty() {
        return Vec::new(); // `split` on an empty slice yields one empty group
    }
    let mut groups: Vec<Vec<i32>> = index.split(|&i| i < 0).map(<[i32]>::to_vec).collect();
    // `split` yields a trailing empty group after a final -1; drop just that one.
    if index.last().is_some_and(|&i| i < 0) {
        groups.pop();
    }
    groups
}

/// Look up `values[i]` for an X3D (signed) index, with a helpful error.
fn get<T: Copy>(values: &[T], i: i64, what: &'static str) -> Result<T, IndexError> {
    usize::try_from(i).ok().and_then(|u| values.get(u).copied()).ok_or(IndexError { what, index: i, len: values.len() })
}

/// Resolve X3D's colour/normal binding rules into a [`Binding`].
///
/// - `per_vertex` + `index` given → corners use `index` (split at -1).
/// - `per_vertex`, no `index` → corners use the coordinate index.
/// - per face + `index` given → face *f* uses `index[f]`.
/// - per face, no `index` → face *f* uses value *f*.
pub fn bind<T: Copy>(
    values: &[T],
    per_vertex: bool,
    index: &[i32],
    faces: &[Vec<u32>],
    what: &'static str,
) -> Result<Binding<T>, IndexError> {
    if values.is_empty() {
        return Ok(Binding::None);
    }
    if per_vertex {
        let corners = if index.is_empty() {
            faces
                .iter()
                .map(|f| f.iter().map(|&c| get(values, i64::from(c), what)).collect())
                .collect::<Result<Vec<Vec<T>>, _>>()?
        } else {
            let groups = split_at_minus_one(index);
            faces
                .iter()
                .enumerate()
                .map(|(fi, face)| {
                    let g = groups.get(fi).map_or(&[][..], Vec::as_slice);
                    (0..face.len())
                        .map(|k| {
                            let i = g.get(k).copied().unwrap_or(-1);
                            get(values, i64::from(i), what)
                        })
                        .collect()
                })
                .collect::<Result<Vec<Vec<T>>, _>>()?
        };
        Ok(Binding::PerCorner(corners))
    } else {
        let per_face = (0..faces.len())
            .map(|f| {
                let i = if index.is_empty() { f as i64 } else { i64::from(*index.get(f).unwrap_or(&-1)) };
                get(values, i, what)
            })
            .collect::<Result<_, _>>()?;
        Ok(Binding::PerFace(per_face))
    }
}

/// Polygons before triangulation. Built by the parser, consumed by
/// [`Polygons::tessellate`].
#[derive(Debug, Clone, Default)]
pub struct Polygons {
    pub points: Vec<Vec3>,
    /// Indices into `points`, one list per face. Faces with fewer than 3
    /// corners are kept (for per-face binding) but produce no triangles.
    pub faces: Vec<Vec<u32>>,
    pub colors: Binding<Vec4>,
    pub normals: Binding<Vec3>,
    /// (s, t, 0) coordinates, or positions for `TextureCoordinateGenerator`.
    /// `None` → X3D's default: project onto the bounding box's two longest sides.
    pub texcoords: Binding<Vec3>,
    pub ccw: bool,
    /// Radians. 0 = faceted, ≥ π = fully smooth.
    pub crease_angle: f32,
}

impl Polygons {
    /// Check every coordinate index before we trust it.
    pub fn validate(&self) -> Result<(), IndexError> {
        let len = self.points.len();
        for &c in self.faces.iter().flatten() {
            if c as usize >= len {
                return Err(IndexError { what: "coordinate", index: i64::from(c), len });
            }
        }
        Ok(())
    }

    /// Newell's method: sums edge cross-products, so the result's length is
    /// twice the polygon's area — big faces weigh more when smoothing.
    fn newell(&self, face: &[u32]) -> Vec3 {
        let p = |i: usize| self.points[face[i % face.len()] as usize];
        let n = (0..face.len()).fold(Vec3::ZERO, |acc, i| {
            let (a, b) = (p(i), p(i + 1));
            acc + Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y))
        });
        if self.ccw { n } else { -n }
    }

    /// Triangulate (fan), compute or apply normals, and weld duplicates.
    pub fn tessellate(&self) -> Mesh {
        let face_normals: Vec<Vec3> = self.faces.iter().map(|f| self.newell(f)).collect();

        // For smoothing: which faces touch each coordinate?
        let smooth = self.crease_angle > 1e-4 && matches!(self.normals, Binding::None);
        let mut faces_at: Vec<Vec<usize>> = Vec::new();
        if smooth {
            faces_at = vec![Vec::new(); self.points.len()];
            for (fi, face) in self.faces.iter().enumerate() {
                for &c in face {
                    faces_at[c as usize].push(fi);
                }
            }
        }
        let cos_crease = self.crease_angle.min(PI).cos() - 1e-5;

        let corner_normal = |fi: usize, k: usize, c: u32| -> Vec3 {
            let n = match &self.normals {
                Binding::PerCorner(v) => v[fi][k],
                Binding::PerFace(v) => v[fi],
                Binding::None if smooth => {
                    let nf = face_normals[fi].normalize_or_zero();
                    faces_at[c as usize]
                        .iter()
                        .map(|&g| face_normals[g])
                        .filter(|ng| nf.dot(ng.normalize_or_zero()) >= cos_crease)
                        .sum()
                }
                Binding::None => face_normals[fi],
            };
            n.try_normalize().unwrap_or(Vec3::Y)
        };
        let corner_color = |fi: usize, k: usize| -> Vec4 {
            match &self.colors {
                Binding::PerCorner(v) => v[fi][k],
                Binding::PerFace(v) => v[fi],
                Binding::None => Vec4::ONE,
            }
        };

        // X3D's default texture mapping: S along the box's longest side, T
        // along the second longest, both scaled by the longest.
        let bounds: crate::Aabb = self.points.iter().copied().collect();
        let size = bounds.size();
        let mut axes = [0usize, 1, 2];
        axes.sort_by(|&a, &b| size[b].total_cmp(&size[a]));
        let (s_axis, t_axis, scale) = (axes[0], axes[1], size[axes[0]].max(1e-12));
        let corner_tex = |fi: usize, k: usize, c: u32| -> Vec3 {
            match &self.texcoords {
                Binding::PerCorner(v) => v[fi][k],
                Binding::PerFace(v) => v[fi],
                Binding::None => {
                    let p = self.points[c as usize] - bounds.min;
                    Vec3::new(p[s_axis] / scale, p[t_axis] / scale, 0.0)
                }
            }
        };

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (fi, face) in self.faces.iter().enumerate() {
            if face.len() < 3 {
                continue;
            }
            let base = vertices.len() as u32;
            for (k, &c) in face.iter().enumerate() {
                vertices.push(Vertex {
                    position: self.points[c as usize].into(),
                    normal: corner_normal(fi, k, c).into(),
                    color: corner_color(fi, k).into(),
                    texcoord: corner_tex(fi, k, c).into(),
                });
            }
            for k in 1..face.len() as u32 - 1 {
                // ccw=false means the file lists corners clockwise: flip.
                let (b, c) = if self.ccw { (k, k + 1) } else { (k + 1, k) };
                indices.extend([base, base + b, base + c]);
            }
        }
        let mut mesh = Mesh::new(Topology::Triangles, vertices, indices);
        mesh.has_colors = !matches!(self.colors, Binding::None);
        mesh.weld();
        mesh
    }
}

/// Polylines (IndexedLineSet / LineSet) → a line *list* (pairs of indices).
/// `colors` is per-corner (per vertex) or per-face (per polyline).
pub fn polylines(points: &[Vec3], lines: &[Vec<u32>], colors: &Binding<Vec4>) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (li, line) in lines.iter().enumerate() {
        if line.len() < 2 {
            continue;
        }
        let base = vertices.len() as u32;
        for (k, &c) in line.iter().enumerate() {
            let color = match colors {
                Binding::PerCorner(v) => v[li][k],
                Binding::PerFace(v) => v[li],
                Binding::None => Vec4::ONE,
            };
            vertices.push(Vertex {
                position: points[c as usize].into(),
                normal: [0.0; 3],
                color: color.into(),
                texcoord: [0.0; 3],
            });
        }
        // 🦀 `windows(2)` would work on a slice; here a range does the same.
        for k in 0..line.len() as u32 - 1 {
            indices.extend([base + k, base + k + 1]);
        }
    }
    let mut mesh = Mesh::new(Topology::Lines, vertices, indices);
    mesh.has_colors = !matches!(colors, Binding::None);
    mesh.solid = false;
    mesh
}

/// PointSet → a point list.
pub fn point_set(points: &[Vec3], colors: Option<&[Vec4]>) -> Mesh {
    let vertices: Vec<Vertex> = points
        .iter()
        .enumerate()
        .map(|(i, p)| Vertex {
            position: (*p).into(),
            normal: [0.0; 3],
            color: colors.and_then(|c| c.get(i)).copied().unwrap_or(Vec4::ONE).into(),
            texcoord: [0.0; 3],
        })
        .collect();
    let indices = (0..vertices.len() as u32).collect();
    let mut mesh = Mesh::new(Topology::Points, vertices, indices);
    mesh.has_colors = colors.is_some();
    mesh.solid = false;
    mesh
}

// ---------------------------------------------------------------------------
// Primitives: Box, Sphere, Cylinder, Cone
// ---------------------------------------------------------------------------

/// A small helper that builds indexed triangles and keeps every triangle
/// facing the way its first vertex's normal points (so we can never get the
/// winding wrong by hand).
#[derive(Default)]
struct Builder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Builder {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2) -> u32 {
        self.vertices.push(Vertex {
            position: p.into(),
            normal: n.into(),
            color: [1.0; 4],
            texcoord: uv.extend(0.0).into(),
        });
        (self.vertices.len() - 1) as u32
    }
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        let pos = |i: u32| Vec3::from(self.vertices[i as usize].position);
        let nrm = |i: u32| Vec3::from(self.vertices[i as usize].normal);
        let geometric = (pos(b) - pos(a)).cross(pos(c) - pos(a));
        let wanted = nrm(a) + nrm(b) + nrm(c);
        if geometric.dot(wanted) >= 0.0 {
            self.indices.extend([a, b, c]);
        } else {
            self.indices.extend([a, c, b]);
        }
    }
    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.tri(a, b, c);
        self.tri(a, c, d);
    }
    fn finish(self) -> Mesh {
        Mesh::new(Topology::Triangles, self.vertices, self.indices)
    }
}

/// X3D `Box`: centred on the origin, `size` = full edge lengths.
pub fn cuboid(size: Vec3) -> Mesh {
    let h = size * 0.5;
    let mut b = Builder::default();
    // Each face: its normal and two in-plane axes.
    for n in [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z] {
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let c = n * h;
        let corner = |su: f32, sv: f32| c + (u * su + v * sv) * h;
        let i = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]
            .map(|(su, sv)| b.vertex(corner(su, sv), n, Vec2::new(su + 1.0, sv + 1.0) * 0.5));
        b.quad(i[0], i[1], i[2], i[3]);
    }
    b.finish()
}

/// X3D `Sphere`: a UV sphere with `segments` around and `segments/2` rings.
pub fn sphere(radius: f32, segments: u32) -> Mesh {
    let (seg, rings) = (segments.max(3), (segments / 2).max(2));
    let mut b = Builder::default();
    let mut grid = Vec::with_capacity(((rings + 1) * (seg + 1)) as usize);
    for i in 0..=rings {
        let theta = PI * i as f32 / rings as f32; // 0 at the north pole
        for j in 0..=seg {
            let phi = TAU * j as f32 / seg as f32;
            let n = Vec3::new(theta.sin() * phi.sin(), theta.cos(), theta.sin() * phi.cos());
            // X3D: s wraps from the back (−Z), t runs 0 (bottom) to 1 (top).
            let uv = Vec2::new(j as f32 / seg as f32 + 0.5, 1.0 - i as f32 / rings as f32);
            grid.push(b.vertex(n * radius, n, uv));
        }
    }
    let at = |i: u32, j: u32| grid[(i * (seg + 1) + j) as usize];
    for i in 0..rings {
        for j in 0..seg {
            let (a, bb, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            // Skip the zero-area triangles that touch a pole.
            if i != rings - 1 {
                b.tri(a, bb, c);
            }
            if i != 0 {
                b.tri(a, c, d);
            }
        }
    }
    b.finish()
}

/// Which parts of a Cylinder/Cone to build.
#[derive(Debug, Clone, Copy)]
pub struct Parts {
    pub side: bool,
    pub top: bool,
    pub bottom: bool,
}

fn disk(b: &mut Builder, y: f32, radius: f32, normal: Vec3, segments: u32) {
    let center = b.vertex(Vec3::new(0.0, y, 0.0), normal, Vec2::splat(0.5));
    let ring: Vec<u32> = (0..=segments)
        .map(|j| {
            let phi = TAU * j as f32 / segments as f32;
            let (x, z) = (phi.sin(), phi.cos());
            b.vertex(Vec3::new(radius * x, y, radius * z), normal, Vec2::new(x * 0.5 + 0.5, 0.5 - z * 0.5))
        })
        .collect();
    for w in ring.windows(2) {
        b.tri(center, w[0], w[1]);
    }
}

/// X3D `Cylinder`: axis along Y, centred on the origin.
pub fn cylinder(radius: f32, height: f32, parts: Parts, segments: u32) -> Mesh {
    let seg = segments.max(3);
    let (y0, y1) = (-height / 2.0, height / 2.0);
    let mut b = Builder::default();
    if parts.side {
        let cols: Vec<(u32, u32)> = (0..=seg)
            .map(|j| {
                let phi = TAU * j as f32 / seg as f32;
                let n = Vec3::new(phi.sin(), 0.0, phi.cos());
                let s = j as f32 / seg as f32 + 0.5;
                (
                    b.vertex(n * radius + Vec3::Y * y1, n, Vec2::new(s, 1.0)),
                    b.vertex(n * radius + Vec3::Y * y0, n, Vec2::new(s, 0.0)),
                )
            })
            .collect();
        for w in cols.windows(2) {
            let ((t0, b0), (t1, b1)) = (w[0], w[1]);
            b.quad(t0, b0, b1, t1);
        }
    }
    if parts.top {
        disk(&mut b, y1, radius, Vec3::Y, seg);
    }
    if parts.bottom {
        disk(&mut b, y0, radius, Vec3::NEG_Y, seg);
    }
    b.finish()
}

/// X3D `Cone`: apex at +height/2, base at −height/2.
pub fn cone(bottom_radius: f32, height: f32, parts: Parts, segments: u32) -> Mesh {
    let seg = segments.max(3);
    let (y0, y1) = (-height / 2.0, height / 2.0);
    let mut b = Builder::default();
    if parts.side {
        // The side normal tilts up by the cone's slope.
        let normal_at = |phi: f32| Vec3::new(phi.sin() * height, bottom_radius, phi.cos() * height).normalize();
        for j in 0..seg {
            let (p0, p1) = (TAU * j as f32 / seg as f32, TAU * (j + 1) as f32 / seg as f32);
            let base = |phi: f32| Vec3::new(bottom_radius * phi.sin(), y0, bottom_radius * phi.cos());
            let s = |j: u32| j as f32 / seg as f32 + 0.5;
            let a = b.vertex(base(p0), normal_at(p0), Vec2::new(s(j), 0.0));
            let c = b.vertex(base(p1), normal_at(p1), Vec2::new(s(j + 1), 0.0));
            // One apex vertex per segment, with the in-between normal, so
            // the tip shades smoothly instead of pinching.
            let apex = b.vertex(Vec3::Y * y1, normal_at((p0 + p1) / 2.0), Vec2::new((s(j) + s(j + 1)) / 2.0, 1.0));
            b.tri(a, c, apex);
        }
    }
    if parts.bottom {
        disk(&mut b, y0, bottom_radius, Vec3::NEG_Y, seg);
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every triangle must face the same way as its vertex normals.
    fn assert_outward(m: &Mesh) {
        for t in m.indices.chunks_exact(3) {
            let v = |i: u32| m.vertices[i as usize];
            let p = |i: u32| Vec3::from(v(i).position);
            let geometric = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            if geometric.length() < 1e-9 {
                continue;
            }
            let n = Vec3::from(v(t[0]).normal);
            assert!(geometric.dot(n) > 0.0, "triangle {t:?} faces inward in {}", m.label);
        }
    }

    #[test]
    fn primitives_face_outward() {
        let all = Parts { side: true, top: true, bottom: true };
        for m in [cuboid(Vec3::splat(2.0)), sphere(1.0, 24), cylinder(1.0, 2.0, all, 24), cone(1.0, 2.0, all, 24)] {
            assert!(!m.indices.is_empty());
            assert_outward(&m);
        }
    }

    #[test]
    fn box_has_twelve_triangles() {
        assert_eq!(cuboid(Vec3::ONE).primitive_count(), 12);
    }

    #[test]
    fn split_handles_trailing_minus_one() {
        assert_eq!(split_at_minus_one(&[0, 1, 2, -1]), [vec![0, 1, 2]]);
        assert_eq!(split_at_minus_one(&[0, 1, 2, -1, 3, 4, 5]), [vec![0, 1, 2], vec![3, 4, 5]]);
        assert_eq!(split_at_minus_one(&[]).len(), 0);
    }

    fn unit_square() -> Polygons {
        Polygons {
            points: vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y],
            faces: vec![vec![0, 1, 2, 3]],
            ccw: true,
            ..Default::default()
        }
    }

    #[test]
    fn quad_fans_into_two_triangles_facing_z() {
        let m = unit_square().tessellate();
        assert_eq!(m.primitive_count(), 2);
        assert!(m.vertices.iter().all(|v| v.normal == [0.0, 0.0, 1.0]));
        assert_outward(&m);
    }

    #[test]
    fn clockwise_faces_are_flipped() {
        let mut p = unit_square();
        p.ccw = false;
        let m = p.tessellate();
        assert!(m.vertices.iter().all(|v| v.normal == [0.0, 0.0, -1.0]));
        assert_outward(&m);
    }

    #[test]
    fn crease_angle_smooths_a_shallow_fold() {
        // Two quads folded by 20°: smooth with creaseAngle 0.5 rad, faceted with 0.
        let a = 20f32.to_radians();
        let mut p = Polygons {
            points: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(2.0, 0.0, -a.sin()),
                Vec3::new(2.0, 1.0, -a.sin()),
            ],
            faces: vec![vec![0, 1, 2, 3], vec![1, 4, 5, 2]],
            ccw: true,
            ..Default::default()
        };
        let faceted = p.tessellate();
        p.crease_angle = 0.5;
        let smooth = p.tessellate();
        // Smooth shading shares the two seam vertices → fewer unique vertices.
        assert!(smooth.vertices.len() < faceted.vertices.len());
    }

    #[test]
    fn bad_index_is_reported() {
        let mut p = unit_square();
        p.faces[0][2] = 99;
        assert_eq!(p.validate().unwrap_err().index, 99);
    }

    #[test]
    fn per_face_colours_use_colour_index() {
        let faces = vec![vec![0, 1, 2], vec![2, 3, 0]];
        let red_green = [Vec4::X, Vec4::Y];
        let b = bind(&red_green, false, &[1, 0], &faces, "color").unwrap();
        assert_eq!(b, Binding::PerFace(vec![Vec4::Y, Vec4::X]));
        assert!(bind(&red_green, false, &[5, 0], &faces, "color").is_err());
    }
}
