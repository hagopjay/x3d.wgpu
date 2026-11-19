//! X3D ComposedShader GLSL → GLSL 4.50 that naga (wgpu's shader compiler) accepts.
//!
//! X3D browsers run WebGL shaders: GLSL ES (`#version 300 es`, or the older
//! 1.00 with `attribute`/`varying`), loose `uniform float time;` variables,
//! and the X_ITE naming conventions (`x3d_Vertex`, `x3d_ModelViewMatrix`,
//! `x3d_FragColor`…). Vulkan-style GLSL, which naga reads, wants instead:
//!
//! - every loose uniform inside one **uniform block** with a set/binding;
//! - an explicit `layout(location = N)` on every `in` and `out`, matching by
//!   number between the stages (WebGL matched them by *name*);
//! - no `precision` statements, and `#version 450`.
//!
//! 🎨 **Point sprites.** WebGL draws a point as a square `gl_PointSize`
//! pixels wide and tells the fragment shader where it is inside it
//! (`gl_PointCoord`). WebGPU has neither. So for a PointSet we rename the
//! author's `main` to `x3d_user_main`, call it from a new `main` that grows
//! each point into a 6-vertex quad (one instance per point), and pass the
//! corner through a varying that stands in for `gl_PointCoord`.
//!
//! 🦀 This module is plain text processing with no GPU, so it has thorough
//! unit tests; `custom.rs` hands the result to naga.

use std::collections::HashMap;
use std::fmt::Write;

/// Vertex attributes we can feed, by X_ITE name → `@location`.
/// (`Vertex` in x3d-scene: position, normal, colour, texcoord.)
pub const ATTRIBUTES: &[(&str, u32)] = &[
    ("x3d_Vertex", 0),
    ("x3d_Normal", 1),
    ("x3d_Color", 2),
    ("x3d_TexCoord0", 3),
    ("x3d_TexCoord", 3),
    ("x3d_MultiTexCoord0", 3),
];

/// Uniform types allowed in the block (no samplers or arrays yet).
const UNIFORM_TYPES: &[&str] =
    &["float", "int", "uint", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "mat2", "mat3", "mat4"];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GlslError {
    #[error("vertex attribute `{0}` isn't one X3D provides (x3d_Vertex, x3d_Normal, x3d_Color, x3d_TexCoord0)")]
    UnknownAttribute(String),
    #[error("uniform `{name}` is `{first}` in one stage and `{second}` in the other")]
    TypeClash { name: String, first: String, second: String },
    #[error("the fragment shader reads `{0}`, which the vertex shader never writes")]
    MissingVarying(String),
    #[error("not supported yet: {0}")]
    Unsupported(String),
}

/// The two translated stages, and the uniforms they share (in block order).
#[derive(Debug, Clone, PartialEq)]
pub struct Translated {
    pub vertex: String,
    pub fragment: String,
    /// `(type, name)` of every block member.
    pub uniforms: Vec<(String, String)>,
}

/// One GLSL declaration we lift out of the source.
#[derive(Debug, Clone, PartialEq)]
struct Decl {
    qualifiers: Vec<String>, // flat, smooth, …
    ty: String,
    name: String,
}

/// What a stage's source turns into once declarations are lifted out.
#[derive(Debug, Default)]
struct Stage {
    uniforms: Vec<Decl>,
    inputs: Vec<Decl>,
    outputs: Vec<Decl>,
    body: String,
}

const PRECISIONS: &[&str] = &["lowp", "mediump", "highp"];
const INTERPOLATION: &[&str] = &["flat", "smooth", "noperspective", "centroid"];

/// Split one declaration line (`flat out highp vec3 a, b;`) after its storage
/// keyword into declarations.
fn parse_decl(rest: &str) -> Option<Vec<Decl>> {
    let rest = rest.trim().strip_suffix(';')?;
    let mut words = rest.split_whitespace().peekable();
    let mut qualifiers = Vec::new();
    while let Some(w) = words.peek() {
        if INTERPOLATION.contains(w) {
            qualifiers.push((*w).to_owned());
            words.next();
        } else if PRECISIONS.contains(w) {
            words.next();
        } else {
            break;
        }
    }
    let ty = words.next()?.to_owned();
    let names: String = words.collect::<Vec<_>>().join(" ");
    Some(
        names
            .split(',')
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .map(|name| Decl { qualifiers: qualifiers.clone(), ty: ty.clone(), name: name.to_owned() })
            .collect(),
    )
}

/// Replace whole identifiers only (`gl_PointSize`, not `my_gl_PointSize2`).
fn replace_word(src: &str, from: &str, to: &str) -> String {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(i) = rest.find(from) {
        let before_ok = rest[..i].chars().next_back().is_none_or(|c| !is_ident(c));
        let after_ok = rest[i + from.len()..].chars().next().is_none_or(|c| !is_ident(c));
        out.push_str(&rest[..i]);
        out.push_str(if before_ok && after_ok { to } else { from });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

/// Put every leading declaration on a line of its own, so that minified
/// sources (`in vec4 p; out vec2 uv; void main(){…}`) split like tidy ones.
fn declarations_first(src: &str) -> String {
    const STARTS: &[&str] = &["uniform ", "in ", "out ", "attribute ", "varying ", "precision ", "flat ", "smooth "];
    let mut out = String::with_capacity(src.len() + 16);
    for line in src.lines() {
        let mut rest = line.trim_start();
        while STARTS.iter().any(|k| rest.starts_with(k)) {
            match rest.find(';') {
                Some(i) => {
                    out.push_str(&rest[..=i]);
                    out.push('\n');
                    rest = rest[i + 1..].trim_start();
                }
                None => break,
            }
        }
        out.push_str(rest);
        out.push('\n');
    }
    out
}

/// Lift uniforms and in/out declarations out of one stage.
fn split_stage(src: &str, vertex: bool) -> Result<Stage, GlslError> {
    let mut stage = Stage::default();
    for line in declarations_first(src).lines() {
        let t = line.trim_start();
        let first = t.split_whitespace().next().unwrap_or("");
        // A declaration may start with an interpolation qualifier: `flat out …`.
        let (storage, rest) = if INTERPOLATION.contains(&first) {
            let after = t[first.len()..].trim_start();
            let kw = after.split_whitespace().next().unwrap_or("");
            (kw, format!("{first} {}", &after[kw.len()..]))
        } else {
            (first, t[first.len()..].to_owned())
        };
        let list = match storage {
            "#version" | "precision" => continue,
            "uniform" | "in" | "out" | "attribute" | "varying" if t.trim_end().ends_with(';') => {
                parse_decl(&rest).unwrap_or_default()
            }
            _ => {
                stage.body.push_str(line);
                stage.body.push('\n');
                continue;
            }
        };
        match (storage, vertex) {
            ("uniform", _) => stage.uniforms.extend(list),
            ("in" | "attribute", true) | ("in" | "varying", false) => stage.inputs.extend(list),
            ("out" | "varying", true) | ("out", false) => stage.outputs.extend(list),
            _ => unreachable!("storage keywords are matched above"),
        }
    }
    for u in &stage.uniforms {
        if u.name.contains('[') {
            return Err(GlslError::Unsupported(format!("uniform arrays (`{}`)", u.name)));
        }
        if !UNIFORM_TYPES.contains(&u.ty.as_str()) {
            return Err(GlslError::Unsupported(format!("uniform `{} {}` (samplers and structs)", u.ty, u.name)));
        }
    }
    Ok(stage)
}

/// Translate one ComposedShader. `sprites`: the shape is a PointSet.
pub fn translate(vertex: &str, fragment: &str, sprites: bool) -> Result<Translated, GlslError> {
    let mut vs = split_stage(vertex, true)?;
    let mut fs = split_stage(fragment, false)?;

    // GLSL ES 1.00 leftovers.
    for body in [&mut vs.body, &mut fs.body] {
        *body = replace_word(body, "texture2D", "texture");
        *body = replace_word(body, "textureCube", "texture");
    }
    if fs.outputs.is_empty() {
        fs.body = replace_word(&fs.body, "gl_FragColor", "x3d_FragColor");
        fs.outputs.push(Decl { qualifiers: vec![], ty: "vec4".into(), name: "x3d_FragColor".into() });
    }

    // One block for both stages: union of their uniforms, in order of appearance.
    let mut uniforms: Vec<(String, String)> = Vec::new();
    for u in vs.uniforms.iter().chain(&fs.uniforms) {
        match uniforms.iter().find(|(_, n)| *n == u.name) {
            Some((ty, _)) if *ty != u.ty => {
                return Err(GlslError::TypeClash { name: u.name.clone(), first: ty.clone(), second: u.ty.clone() });
            }
            Some(_) => {}
            None => uniforms.push((u.ty.clone(), u.name.clone())),
        }
    }
    if sprites && !uniforms.iter().any(|(_, n)| n == "x3d_Viewport") {
        uniforms.push(("ivec4".into(), "x3d_Viewport".into())); // the sprite quad needs it
    }
    let mut block = String::from("layout(set = 0, binding = 0) uniform X3DUniforms {\n");
    for (ty, name) in &uniforms {
        let _ = writeln!(block, "    {ty} {name};");
    }
    block.push_str("};\n");

    // Varyings: numbered in the vertex shader's order, matched by name.
    if sprites {
        vs.outputs.push(Decl { qualifiers: vec![], ty: "vec2".into(), name: "x3d_PointCoord".into() });
        fs.body = replace_word(&fs.body, "gl_PointCoord", "x3d_PointCoord");
        if fs.body.contains("x3d_PointCoord") {
            fs.inputs.push(Decl { qualifiers: vec![], ty: "vec2".into(), name: "x3d_PointCoord".into() });
        }
    }
    let varying: HashMap<&str, (usize, &Decl)> =
        vs.outputs.iter().enumerate().map(|(i, d)| (d.name.as_str(), (i, d))).collect();

    let mut v = format!("#version 450\n{block}");
    for a in &vs.inputs {
        let loc = ATTRIBUTES
            .iter()
            .find(|(n, _)| *n == a.name)
            .map(|(_, l)| *l)
            .ok_or_else(|| GlslError::UnknownAttribute(a.name.clone()))?;
        let _ = writeln!(v, "layout(location = {loc}) in {} {};", a.ty, a.name);
    }
    for (i, o) in vs.outputs.iter().enumerate() {
        let _ = writeln!(v, "layout(location = {i}) {} out {} {};", o.qualifiers.join(" "), o.ty, o.name);
    }
    let mut f = format!("#version 450\n{block}");
    for input in &fs.inputs {
        let (loc, from_vs) =
            varying.get(input.name.as_str()).ok_or_else(|| GlslError::MissingVarying(input.name.clone()))?;
        let _ =
            writeln!(f, "layout(location = {loc}) {} in {} {};", from_vs.qualifiers.join(" "), input.ty, input.name);
    }
    for (i, o) in fs.outputs.iter().enumerate() {
        let _ = writeln!(f, "layout(location = {i}) out {} {};", o.ty, o.name);
    }

    if sprites {
        let body = replace_word(&vs.body, "gl_PointSize", "x3d_PointSize");
        let body =
            body.replacen("void main(", "void x3d_user_main(", 1).replacen("void main (", "void x3d_user_main(", 1);
        v.push_str("float x3d_PointSize;\n");
        v.push_str(&body);
        v.push_str(
            "
void main() {
    x3d_PointSize = 1.0;
    x3d_user_main();
    // Grow the point into a square of x3d_PointSize pixels (two triangles).
    vec2 corners[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
                              vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    vec2 corner = corners[gl_VertexIndex % 6];
    vec2 pixels = vec2(float(x3d_Viewport.z), float(x3d_Viewport.w));
    gl_Position = vec4(gl_Position.xy + corner * x3d_PointSize / pixels * gl_Position.w, gl_Position.zw);
    // gl_PointCoord: (0, 0) at the top-left of the square.
    x3d_PointCoord = vec2(corner.x, -corner.y) * 0.5 + 0.5;
}
",
        );
    } else {
        v.push_str(&vs.body);
    }
    f.push_str(&fs.body);
    Ok(Translated { vertex: v, fragment: f, uniforms })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VS: &str = "#version 300 es
precision highp float;
uniform mat4 x3d_ProjectionMatrix;
uniform mat4 x3d_ModelViewMatrix;
uniform float time, phase;
in vec4 x3d_Vertex;
in vec4 x3d_Color;
out vec4 vColor;
flat out float vKind;
void main() {
  vColor = x3d_Color * phase;
  vKind = 1.0;
  gl_PointSize = 4.0 + time;
  gl_Position = x3d_ProjectionMatrix * x3d_ModelViewMatrix * x3d_Vertex;
}";
    const FS: &str = "#version 300 es
precision mediump float;
uniform float time;
flat in float vKind;
in vec4 vColor;
out vec4 x3d_FragColor;
void main() { x3d_FragColor = vColor * vKind * gl_PointCoord.x; }";

    #[test]
    fn uniforms_become_one_block_shared_by_both_stages() {
        let t = translate(VS, FS, false).unwrap();
        let names: Vec<&str> = t.uniforms.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["x3d_ProjectionMatrix", "x3d_ModelViewMatrix", "time", "phase"]);
        assert!(t.vertex.starts_with("#version 450\nlayout(set = 0, binding = 0) uniform X3DUniforms {"));
        assert!(!t.vertex.contains("precision") && !t.fragment.contains("uniform float time;"));
    }

    #[test]
    fn varyings_match_by_name_not_order() {
        let t = translate(VS, FS, false).unwrap();
        assert!(t.vertex.contains("layout(location = 0)  out vec4 vColor;"), "{}", t.vertex);
        assert!(t.vertex.contains("layout(location = 1) flat out float vKind;"));
        // The fragment shader lists them the other way round; locations follow the VS.
        assert!(t.fragment.contains("layout(location = 1) flat in float vKind;"), "{}", t.fragment);
        assert!(t.fragment.contains("layout(location = 0)  in vec4 vColor;"));
        assert!(t.vertex.contains("layout(location = 2) in vec4 x3d_Color;"));
    }

    #[test]
    fn point_sprites_wrap_main() {
        let t = translate(VS, FS, true).unwrap();
        assert!(t.vertex.contains("void x3d_user_main()"));
        assert!(t.vertex.contains("x3d_PointSize = 4.0 + time;"));
        assert!(t.vertex.contains("out vec2 x3d_PointCoord;"));
        assert!(t.fragment.contains("vKind * x3d_PointCoord.x"));
        assert!(t.uniforms.iter().any(|(_, n)| n == "x3d_Viewport"));
    }

    #[test]
    fn glsl_100_is_upgraded() {
        let t = translate(
            "attribute vec4 x3d_Vertex; varying vec2 uv; void main(){ uv = x3d_Vertex.xy; gl_Position = x3d_Vertex; }",
            "varying vec2 uv; uniform float k; void main(){ gl_FragColor = vec4(uv, k, 1.0); }",
            false,
        )
        .unwrap();
        assert!(t.fragment.contains("layout(location = 0) out vec4 x3d_FragColor;"));
        assert!(t.fragment.contains("x3d_FragColor = vec4(uv, k, 1.0)"));
    }

    #[test]
    fn helpful_errors() {
        let e = translate("in vec4 position; void main(){}", "out vec4 c; void main(){}", false).unwrap_err();
        assert_eq!(e, GlslError::UnknownAttribute("position".into()));
        let e = translate("void main(){}", "in vec3 nowhere; out vec4 c; void main(){}", false).unwrap_err();
        assert_eq!(e, GlslError::MissingVarying("nowhere".into()));
        let e = translate("uniform sampler2D tex; void main(){}", "out vec4 c; void main(){}", false).unwrap_err();
        assert!(matches!(e, GlslError::Unsupported(_)));
    }

    #[test]
    fn whole_words_only() {
        assert_eq!(
            replace_word("gl_PointSize my_gl_PointSize gl_PointSize2", "gl_PointSize", "S"),
            "S my_gl_PointSize gl_PointSize2"
        );
    }
}
