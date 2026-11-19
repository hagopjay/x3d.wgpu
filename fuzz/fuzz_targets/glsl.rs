//! Untrusted ComposedShader text → our GLSL ES 300 → 450 translation →
//! naga parse + validate + reflect. Must never panic: shader text comes
//! from whatever X3D file was opened.
//!
//! Input: vertex source, a 0xFF byte, fragment source, then one flag byte
//! (odd = drawn as point sprites).
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let (body, sprites) = match data.split_last() {
        Some((flag, rest)) => (rest, flag & 1 == 1),
        None => return,
    };
    let mut parts = body.splitn(2, |&b| b == 0xFF);
    let vertex = String::from_utf8_lossy(parts.next().unwrap_or_default()).into_owned();
    let fragment = String::from_utf8_lossy(parts.next().unwrap_or_default()).into_owned();
    let src = x3d_scene::ShaderSource { label: "fuzz".into(), vertex, fragment, fields: Vec::new() };
    let _ = x3d_render::custom::compile(&src, sprites);
});
