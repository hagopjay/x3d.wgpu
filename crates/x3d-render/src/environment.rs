//! Image-based lighting for X3D 4 `EnvironmentLight`.
//!
//! 🎨 **Light from everywhere.** A point light comes from one direction; an
//! environment comes from all of them, stored as a cube map (six images).
//! Shading against it means integrating over the whole sphere, per pixel —
//! far too slow — so we integrate *ahead of time* (the "split sum",
//! Karis 2013):
//!
//! - **Diffuse**: Lambert only needs irradiance per normal, a very smooth
//!   function, so 9 spherical-harmonic coefficients hold it (Ramamoorthi &
//!   Hanrahan 2001). `project_sh9` computes them on the CPU.
//! - **Specular**: a rough mirror sees a blurred environment. We blur the
//!   cube once per roughness step with the GGX lobe and store each blur as
//!   a mip level (`prefilter`); the shader picks the mip by roughness.
//! - The BRDF's remaining factor (Fresnel and shadowing, integrated over
//!   the lobe) is a 2D function of (N·V, roughness); the shader uses Karis's
//!   analytic fit instead of a lookup texture.
//!
//! Everything here is plain CPU code with tests: the GPU only samples.

use glam::{Vec2, Vec3};
use x3d_scene::{CubeMap, Environment, TextureRef};

/// Face size of the prefiltered cube (mip 0).
pub const SIZE: u32 = 64;
/// GGX samples per texel when prefiltering.
const SAMPLES: u32 = 128;

/// A CPU cube map of linear RGB, faces in GPU order (+X −X +Y −Y +Z −Z).
#[derive(Debug, Clone)]
pub struct Cube {
    pub size: u32,
    pub faces: [Vec<Vec3>; 6],
}

/// The direction through texel centre (x, y) of `face` (wgpu/Vulkan convention).
pub fn texel_dir(face: usize, x: u32, y: u32, size: u32) -> Vec3 {
    let u = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
    let v = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
    match face {
        0 => Vec3::new(1.0, -v, -u),
        1 => Vec3::new(-1.0, -v, u),
        2 => Vec3::new(u, 1.0, v),
        3 => Vec3::new(u, -1.0, -v),
        4 => Vec3::new(u, -v, 1.0),
        _ => Vec3::new(-u, -v, -1.0),
    }
    .normalize()
}

/// Inverse of `texel_dir`: (face, u, v) with u, v in 0..1.
pub fn dir_to_face(d: Vec3) -> (usize, f32, f32) {
    let a = d.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if d.x > 0.0 { (0, -d.z, -d.y, a.x) } else { (1, d.z, -d.y, a.x) }
    } else if a.y >= a.z {
        if d.y > 0.0 { (2, d.x, d.z, a.y) } else { (3, d.x, -d.z, a.y) }
    } else if d.z > 0.0 {
        (4, d.x, -d.y, a.z)
    } else {
        (5, -d.x, -d.y, a.z)
    };
    (face, 0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
}

impl Cube {
    pub fn sample(&self, d: Vec3) -> Vec3 {
        let (f, u, v) = dir_to_face(d);
        let n = self.size;
        let x = ((u * n as f32) as u32).min(n - 1);
        let y = ((v * n as f32) as u32).min(n - 1);
        self.faces[f][(y * n + x) as usize]
    }

    /// Decode six images (sRGB bytes → linear), resampled to `size`².
    pub fn load(map: &CubeMap, size: u32) -> Result<Self, String> {
        let mut faces: [Vec<Vec3>; 6] = Default::default();
        for (slot, face) in faces.iter_mut().zip(&map.faces) {
            let img = open(face)?;
            let img = image::imageops::resize(&img, size, size, image::imageops::FilterType::Triangle);
            *slot = img.pixels().map(|p| Vec3::new(srgb(p[0]), srgb(p[1]), srgb(p[2]))).collect();
        }
        Ok(Self { size, faces })
    }
}

fn open(t: &TextureRef) -> Result<image::RgbaImage, String> {
    let img = match &t.inline {
        Some(bytes) => image::load_from_memory(bytes),
        None => image::open(&t.path),
    };
    img.map(|i| i.to_rgba8()).map_err(|e| format!("{}: {e}", t.path.display()))
}

fn srgb(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Solid angle of a cube texel (for weighting integrals over the sphere).
fn texel_solid_angle(x: u32, y: u32, size: u32) -> f32 {
    let area = |u: f32, v: f32| (u * v).atan2((u * u + v * v + 1.0).sqrt());
    let inv = 1.0 / size as f32;
    let (u0, v0) = (2.0 * x as f32 * inv - 1.0, 2.0 * y as f32 * inv - 1.0);
    let (u1, v1) = (u0 + 2.0 * inv, v0 + 2.0 * inv);
    area(u0, v0) - area(u0, v1) - area(u1, v0) + area(u1, v1)
}

/// The 9 real SH basis functions at direction `d`.
fn sh_basis(d: Vec3) -> [f32; 9] {
    let (x, y, z) = (d.x, d.y, d.z);
    [
        0.282_095,
        0.488_603 * y,
        0.488_603 * z,
        0.488_603 * x,
        1.092_548 * x * y,
        1.092_548 * y * z,
        0.315_392 * (3.0 * z * z - 1.0),
        1.092_548 * x * z,
        0.546_274 * (x * x - y * y),
    ]
}

/// Irradiance E(n) as 9 SH coefficients (RGB), already convolved with the
/// cosine lobe: E(n) = Σ cᵢ·Yᵢ(n). For a uniform radiance L, E = π·L.
pub fn project_sh9(cube: &Cube) -> [Vec3; 9] {
    let mut c = [Vec3::ZERO; 9];
    for (f, face) in cube.faces.iter().enumerate() {
        for y in 0..cube.size {
            for x in 0..cube.size {
                let d = texel_dir(f, x, y, cube.size);
                let w = texel_solid_angle(x, y, cube.size);
                let l = face[(y * cube.size + x) as usize];
                for (ci, b) in c.iter_mut().zip(sh_basis(d)) {
                    *ci += l * (b * w);
                }
            }
        }
    }
    // Cosine-lobe convolution per band (Ramamoorthi & Hanrahan 2001).
    use std::f32::consts::{FRAC_PI_4, PI};
    let (b0, b1, b2) = (PI, 2.0 * PI / 3.0, FRAC_PI_4);
    let band = [b0, b1, b1, b1, b2, b2, b2, b2, b2];
    for (ci, a) in c.iter_mut().zip(band) {
        *ci *= a;
    }
    c
}

pub fn eval_sh9(c: &[Vec3; 9], n: Vec3) -> Vec3 {
    c.iter().zip(sh_basis(n)).map(|(ci, b)| *ci * b).sum()
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(i as f32 / n as f32, i.reverse_bits() as f32 * 2.328_306_4e-10)
}

/// A GGX-distributed half vector around +Z, for roughness `r` (α = r²).
fn ggx_sample(xi: Vec2, r: f32) -> Vec3 {
    let a = r * r;
    let phi = std::f32::consts::TAU * xi.x;
    let cos_t = ((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y)).sqrt();
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t)
}

/// The environment as a rough mirror of roughness `r` sees it in direction
/// `n` (with the usual N = V = R assumption).
pub fn prefiltered(cube: &Cube, n: Vec3, r: f32) -> Vec3 {
    if r <= 1e-3 {
        return cube.sample(n);
    }
    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    let (mut sum, mut weight) = (Vec3::ZERO, 0.0);
    for i in 0..SAMPLES {
        let h = ggx_sample(hammersley(i, SAMPLES), r);
        let h = t * h.x + b * h.y + n * h.z;
        let l = 2.0 * n.dot(h) * h - n;
        let nl = n.dot(l);
        if nl > 0.0 {
            sum += cube.sample(l) * nl;
            weight += nl;
        }
    }
    if weight > 0.0 { sum / weight } else { cube.sample(n) }
}

/// Mip levels for a `SIZE` cube: roughness of mip m is m / (levels − 1).
pub fn mip_levels() -> u32 {
    SIZE.ilog2() + 1
}

/// Every mip of the prefiltered specular cube, as RGBA f16 bits per face.
pub fn prefilter(cube: &Cube) -> Vec<[Vec<[u16; 4]>; 6]> {
    let levels = mip_levels();
    (0..levels)
        .map(|m| {
            let size = (SIZE >> m).max(1);
            let r = m as f32 / (levels - 1) as f32;
            let mut faces: [Vec<[u16; 4]>; 6] = Default::default();
            for (f, out) in faces.iter_mut().enumerate() {
                *out = (0..size * size)
                    .map(|i| {
                        let c = prefiltered(cube, texel_dir(f, i % size, i / size, size), r);
                        [f16(c.x), f16(c.y), f16(c.z), f16(1.0)]
                    })
                    .collect();
            }
            faces
        })
        .collect()
}

/// f32 → IEEE half (round to nearest, clamped to the finite range).
pub fn f16(v: f32) -> u16 {
    let v = v.clamp(0.0, 65504.0);
    if v < 6.103_515_6e-5 {
        return (v / 5.960_464_5e-8).round() as u16; // subnormal
    }
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    let rounded = (mant + 0x1000) >> 13;
    ((exp as u32) << 10).wrapping_add(rounded) as u16
}

/// What the renderer uploads for one EnvironmentLight.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub sh: [Vec3; 9],
    pub specular_mips: Vec<[Vec<[u16; 4]>; 6]>,
    pub color_intensity: Vec3,
    pub rotation: glam::Mat4,
}

pub fn prepare(env: &Environment) -> Result<Prepared, String> {
    let diffuse = env.diffuse.as_ref().or(env.specular.as_ref()).ok_or("no cube map")?;
    let specular = env.specular.as_ref().unwrap_or(diffuse);
    let d = Cube::load(diffuse, 32)?;
    let s = if specular == diffuse && SIZE == 32 { d.clone() } else { Cube::load(specular, SIZE)? };
    Ok(Prepared {
        sh: project_sh9(&d),
        specular_mips: prefilter(&s),
        color_intensity: env.color * if env.on { env.intensity } else { 0.0 },
        // Lookups rotate *into* the environment's frame.
        rotation: glam::Mat4::from_quat(env.rotation.inverse()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(l: f32) -> Cube {
        Cube { size: 8, faces: std::array::from_fn(|_| vec![Vec3::splat(l); 64]) }
    }

    #[test]
    fn face_mapping_round_trips() {
        for f in 0..6 {
            for (x, y) in [(0, 0), (3, 5), (7, 7)] {
                let d = texel_dir(f, x, y, 8);
                let (g, u, v) = dir_to_face(d);
                assert_eq!(g, f);
                assert_eq!(((u * 8.0) as u32, (v * 8.0) as u32), (x, y), "face {f}");
            }
        }
    }

    #[test]
    fn solid_angles_cover_the_sphere() {
        let total: f32 = (0..6).map(|_| (0..64).map(|i| texel_solid_angle(i % 8, i / 8, 8)).sum::<f32>()).sum();
        assert!((total - 4.0 * std::f32::consts::PI).abs() < 1e-3, "{total}");
    }

    #[test]
    fn a_uniform_environment_gives_pi_irradiance_everywhere() {
        // White furnace: L = 1 everywhere → E(n) = π for every normal.
        let sh = project_sh9(&uniform(1.0));
        for n in [Vec3::X, Vec3::NEG_Y, Vec3::new(0.3, 0.4, -0.866).normalize()] {
            let e = eval_sh9(&sh, n);
            assert!((e.x - std::f32::consts::PI).abs() < 0.01, "{n}: {e}");
        }
    }

    #[test]
    fn prefiltering_a_uniform_environment_keeps_it_uniform() {
        let c = uniform(0.7);
        for r in [0.0, 0.3, 1.0] {
            let p = prefiltered(&c, Vec3::new(0.2, 0.9, 0.1).normalize(), r);
            assert!((p.x - 0.7).abs() < 1e-4, "roughness {r}: {p}");
        }
    }

    #[test]
    fn half_floats_are_exact_for_simple_values() {
        assert_eq!(f16(1.0), 0x3c00);
        assert_eq!(f16(0.5), 0x3800);
        assert_eq!(f16(0.0), 0);
        assert_eq!(f16(65504.0), 0x7bff);
    }
}
