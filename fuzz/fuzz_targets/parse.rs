//! Any bytes → the X3D loader → the animation runtime. Must never panic:
//! a bad scene is an `Err` or a loss-report entry, not a crash.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let opts = x3d_scene::LoadOptions { segments: 8, ..Default::default() };
    if let Ok(mut scene) = x3d_scene::parse_str_with(&text, &opts) {
        scene.advance(0.0);
        scene.advance(0.75);
        if !scene.viewpoints.is_empty() {
            scene.bind_viewpoint(scene.viewpoints.len() - 1);
        }
        scene.advance(1e6);
        let _ = scene.bounds();
    }
});
