//! The bytes we hand the GPU, mirrored from `shaders/scene.wgsl`.
//!
//! 🎨 **Why everything is a `[f32; 4]`.** WGSL's uniform layout rules align
//! a `vec3f` to 16 bytes, so a Rust `[f32; 3]` followed by another field
//! would *not* line up with the shader. Packing data into `vec4`s (using the
//! spare `w` for a scalar) sidesteps the whole problem — and the `const`
//! assertions below catch any drift at compile time.
//!
//! 🦀 `bytemuck::Pod` promises "any bit pattern is valid and there is no
//! padding", so `bytemuck::bytes_of(&frame)` can safely view the struct as
//! `&[u8]` without copying or `unsafe`.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use x3d_scene::{Light, LightKind, Material};

pub const MAX_LIGHTS: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuLight {
    pub color_intensity: [f32; 4],
    pub position_kind: [f32; 4],
    pub direction_ambient: [f32; 4],
    pub attenuation_radius: [f32; 4],
    pub spot: [f32; 4],
}
const _: () = assert!(size_of::<GpuLight>() == 80);

/// 🦀 `From` conversions make call sites read naturally: `GpuLight::from(&light)`
/// or `light.into()`.
impl From<&Light> for GpuLight {
    fn from(l: &Light) -> Self {
        let (kind, position, direction, attenuation, radius, beam, cut) = match l.kind {
            LightKind::Directional { direction } => {
                (0.0, Vec3::ZERO, direction, Vec3::X, f32::MAX, 0.0, 0.0)
            }
            LightKind::Point {
                location,
                attenuation,
                radius,
            } => (1.0, location, Vec3::NEG_Z, attenuation, radius, 0.0, 0.0),
            LightKind::Spot {
                location,
                direction,
                attenuation,
                radius,
                beam_width,
                cut_off_angle,
            } => (
                2.0,
                location,
                direction,
                attenuation,
                radius,
                beam_width,
                cut_off_angle,
            ),
        };
        Self {
            // A light that is off keeps its slot (draws' light masks index by
            // position) but contributes nothing.
            color_intensity: l.color.extend(if l.on { l.intensity } else { 0.0 }).into(),
            position_kind: position.extend(kind).into(),
            direction_ambient: direction
                .normalize_or(Vec3::NEG_Z)
                .extend(if l.on { l.ambient_intensity } else { 0.0 })
                .into(),
            attenuation_radius: attenuation.extend(radius).into(),
            spot: [beam, cut, 0.0, 0.0],
        }
    }
}

/// Per-frame data: camera + lights. Bound at group 0.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub params: [u32; 4],
    pub near_far: [f32; 4],
    /// Target width and height in pixels, then the line/point size multiplier.
    pub viewport: [f32; 4],
    pub lights: [GpuLight; MAX_LIGHTS],
}
const _: () = assert!(size_of::<FrameUniform>() == 64 + 64 + 16 * 4 + 80 * MAX_LIGHTS);

/// Per-draw data: matrices + material. Bound at group 1 with a dynamic offset.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawUniform {
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub texture_matrix: [[f32; 4]; 4],
    pub diffuse: [f32; 4],
    pub emissive: [f32; 4],
    pub specular: [f32; 4],
    pub pbr: [f32; 4],
    pub flags: [u32; 4],
    /// x: bit *i* set = scene light *i* reaches this draw (X3D light scoping).
    pub light_mask: [u32; 4],
    /// x: line width / point size in pixels.
    pub line: [f32; 4],
}
const _: () = assert!(size_of::<DrawUniform>() == 64 * 3 + 16 * 7);

/// Bits in `flags.w`: which texture slots hold a real texture.
pub mod texture_bits {
    pub const BASE: u32 = 1;
    pub const NORMAL: u32 = 2;
    pub const METAL_ROUGH: u32 = 4;
    pub const OCCLUSION: u32 = 8;
    pub const EMISSIVE: u32 = 16;
}

impl DrawUniform {
    /// `textures`: bits from [`texture_bits`] for the slots that loaded.
    pub fn new(
        model: Mat4,
        material: &Material,
        vertex_colors: bool,
        textures: u32,
        light_mask: u32,
    ) -> Self {
        // 🎨 Normals must be transformed by the inverse-transpose of the
        // model matrix, or non-uniform scale bends them the wrong way.
        let normal_matrix = model.inverse().transpose();
        let pbr = material.pbr.map_or([0.0, 1.0, 1.0, 1.0], |p| {
            [
                p.metallic,
                p.roughness,
                p.occlusion_strength,
                p.normal_scale,
            ]
        });
        Self {
            model: model.to_cols_array_2d(),
            normal_matrix: normal_matrix.to_cols_array_2d(),
            texture_matrix: material.texture_matrix.to_cols_array_2d(),
            diffuse: material.diffuse.extend(1.0 - material.transparency).into(),
            emissive: material.emissive.extend(material.ambient_intensity).into(),
            specular: material.specular.extend(material.shininess).into(),
            pbr,
            flags: [
                u32::from(material.unlit),
                u32::from(vertex_colors),
                u32::from(material.pbr.is_some()),
                textures,
            ],
            light_mask: [light_mask, 0, 0, 0],
            line: [material.line_width, 0.0, 0.0, 0.0],
        }
    }
}
