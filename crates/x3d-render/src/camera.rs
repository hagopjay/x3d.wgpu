//! An orbit camera: it circles a target point, like X3D's EXAMINE mode.
//!
//! 🎨 **The matrices**
//! - *View*: world → camera space (camera at the origin, looking down −Z).
//! - *Projection*: camera space → clip space (perspective divide makes far
//!   things small).
//!
//! 🎨 **Which projection?** wgpu always uses WebGPU's normalized device
//! coordinates — **Y up, depth 0..1** — whatever API runs underneath. In
//! glam's naming that is `proj::directx` (Y up, Z 0..1), *not*
//! `proj::vulkan` (Y down), even though we run on Vulkan: wgpu flips the
//! viewport for us. `proj::opengl` would waste half the depth range (Z −1..1).

use glam::camera::rh::{proj::directx, view};
use glam::{Mat4, Vec3};
use x3d_scene::{Aabb, Viewpoint};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Radians around +Y; 0 looks down −Z (X3D's default view).
    pub yaw: f32,
    /// Radians up from the horizon.
    pub pitch: f32,
    /// X3D field of view: applies to the *smaller* screen dimension.
    pub field_of_view: f32,
    /// Size of the scene, for near/far planes and pan/zoom speed.
    pub scene_radius: f32,
}

impl OrbitCamera {
    /// Frame a bounding box from the front, like X3D's default view.
    pub fn framing(bounds: Aabb) -> Self {
        let (center, radius) = if bounds.is_empty() {
            (Vec3::ZERO, 1.0)
        } else {
            (bounds.center(), (bounds.size().length() * 0.5).max(1e-3))
        };
        let fov = std::f32::consts::FRAC_PI_4;
        Self {
            target: center,
            distance: radius / (fov * 0.5).sin() * 1.1,
            yaw: 0.0,
            pitch: 0.0,
            field_of_view: fov,
            scene_radius: radius,
        }
    }

    /// Start from an X3D Viewpoint. We orbit around its centerOfRotation if
    /// that lies in front of the camera, else around the scene's centre.
    /// (Roll around the view axis is dropped: an orbit camera keeps +Y up.)
    pub fn from_viewpoint(vp: &Viewpoint, bounds: Aabb) -> Self {
        let mut cam = Self::framing(bounds);
        let forward = vp.forward();
        let pick = |p: Vec3| (p - vp.position).dot(forward);
        let mut along = pick(vp.center_of_rotation);
        if along < cam.scene_radius * 0.05 && !bounds.is_empty() {
            along = pick(bounds.center());
        }
        if along <= 1e-3 {
            along = cam.scene_radius.max(1.0);
        }
        let back = -forward; // from target towards the eye
        cam.target = vp.position + forward * along;
        cam.distance = along;
        cam.yaw = back.x.atan2(back.z);
        cam.pitch = back.y.clamp(-1.0, 1.0).asin();
        cam.field_of_view = vp.field_of_view;
        cam
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target + Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    pub fn forward(&self) -> Vec3 {
        (self.target - self.eye()).normalize_or(Vec3::NEG_Z)
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Vertical field of view for this aspect ratio (width / height).
    pub fn fov_y(&self, aspect: f32) -> f32 {
        if aspect >= 1.0 {
            self.field_of_view
        } else {
            // The field of view is horizontal on a tall screen.
            2.0 * ((self.field_of_view * 0.5).tan() / aspect).atan()
        }
    }

    pub fn near_far(&self) -> (f32, f32) {
        let r = self.scene_radius;
        let far = self.distance + r * 4.0;
        let near = (self.distance - r * 2.0).max(far * 1e-4);
        (near, far)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.near_far();
        directx::perspective(self.fov_y(aspect), aspect, near, far)
    }

    // --- input -------------------------------------------------------------

    /// Drag to orbit. `dx`, `dy` in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }

    /// Drag to slide the target across the screen.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let view = self.view();
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        let per_pixel = 2.0 * self.distance * (self.field_of_view * 0.5).tan() / viewport_height.max(1.0);
        self.target += (-right * dx + up * dy) * per_pixel;
    }

    /// Scroll to dolly in and out (multiplicative, so it feels the same at any scale).
    pub fn zoom(&mut self, steps: f32) {
        self.distance = (self.distance * 0.9f32.powf(steps)).max(self.scene_radius * 1e-3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_viewpoint_round_trips() {
        let cam = OrbitCamera::from_viewpoint(&Viewpoint::default(), Aabb::EMPTY);
        assert!((cam.eye() - Vec3::new(0.0, 0.0, 10.0)).length() < 1e-4, "{}", cam.eye());
        assert!((cam.forward() - Vec3::NEG_Z).length() < 1e-5);
    }

    #[test]
    fn turned_viewpoint() {
        // Looking along −X from x = 5.
        let vp = Viewpoint {
            position: Vec3::new(5.0, 0.0, 0.0),
            orientation: glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Default::default()
        };
        let cam = OrbitCamera::from_viewpoint(&vp, Aabb { min: Vec3::splat(-1.0), max: Vec3::ONE });
        assert!((cam.eye() - vp.position).length() < 1e-4);
        assert!((cam.forward() - Vec3::NEG_X).length() < 1e-4);
    }

    #[test]
    fn tall_screens_widen_the_vertical_fov() {
        let cam = OrbitCamera::framing(Aabb::EMPTY);
        assert!(cam.fov_y(0.5) > cam.fov_y(2.0));
    }
}
