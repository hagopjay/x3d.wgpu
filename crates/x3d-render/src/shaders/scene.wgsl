// scene.wgsl — the one shader every X3D shape is drawn with.
//
// 🎨 Try this: run `x3d-view view scenes/05_still_life.x3d --shader crates/x3d-render/src/shaders/scene.wgsl`
// and edit this file while the window is open. Save → it hot-reloads.
// A typo is reported in the terminal and the old shader keeps running.
//
// Layout of the data (must match src/uniforms.rs byte for byte):
//   group 0 = per-frame (camera, lights)       — bound once per frame
//   group 1 = per-draw  (matrices, material)   — one slice of a big buffer,
//                                                 picked with a dynamic offset
//   group 2 = the material's five textures + one sampler

const PI: f32 = 3.14159265;

struct Light {
    color_intensity: vec4f,     // rgb colour, w = intensity
    position_kind: vec4f,       // xyz world location, w = kind (0 dir, 1 point, 2 spot)
    direction_ambient: vec4f,   // xyz world direction, w = ambientIntensity
    attenuation_radius: vec4f,  // xyz = (constant, linear, quadratic), w = radius
    spot: vec4f,                // x = beamWidth, y = cutOffAngle (radians),
                                // z = first shadow matrix (−1 = no shadow), w = shadowIntensity
}

struct Frame {
    view_proj: mat4x4f,
    view: mat4x4f,
    camera_pos: vec4f,          // xyz, w = exposure (linear multiplier for PBR)
    // x = light count, y = debug view, z = colour mode (0 display, 1 linear),
    // w = bit 0: the render target is an *sRGB* format (hardware encodes on write)
    //     bit 1: lights[0] is the headlight (it reaches every draw)
    params: vec4u,
    near_far: vec4f,            // near, far planes; z..w = depth range shown by the Depth view
    viewport: vec4f,            // width, height (pixels), line/point size multiplier
    shadow: vec4f,              // sun light index (−1 = none), cascade count, depth bias, sun shadowIntensity
    cascade_far: vec4f,         // view-space distance where cascade i ends
    cascade_texel: vec4f,       // one texel of cascade i, in world units (for the normal offset)
    shadow_maps: vec4f,         // 1 / sun map size, 1 / local map size
    shadow_mats: array<mat4x4f, 20>, // 0–3 sun cascades, 4–19 spot/point layers (world → light clip)
    env: vec4f,                 // EnvironmentLight: on, specular mips, model (0 single, 1 split, 2 multi-scatter), —
    env_color: vec4f,           // color × intensity
    env_rotation: mat4x4f,      // world → environment
    env_sh: array<vec4f, 9>,    // irradiance SH (cosine-convolved)
    fog: vec4f,                 // x = 0 none, 1 linear, 2 exponential; y = visibilityRange
    fog_color: vec4f,
    lights: array<Light, 16>,
}

struct Draw {
    model: mat4x4f,
    normal_matrix: mat4x4f,     // inverse-transpose of model (handles non-uniform scale)
    texture_matrix: mat4x4f,    // X3D TextureTransform(3D/Matrix3D)
    diffuse: vec4f,             // rgb (diffuseColor or baseColor), a = 1 − transparency
    emissive: vec4f,            // rgb, w = material ambientIntensity
    specular: vec4f,            // rgb, w = shininess (0..1)
    pbr: vec4f,                 // metallic, roughness, occlusionStrength, normalScale
    flags: vec4u,               // x = unlit, y = vertex colours, z = 1 for PhysicalMaterial, w = texture bits
    light_mask: vec4u,          // x: bit i = scene light i reaches this draw (X3D light scoping)
    line: vec4f,                // x: line width / point size in pixels
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var sun_shadows: texture_depth_2d_array;
// A *comparison* sampler: it answers "is the stored depth ≥ mine?" and,
// with linear filtering, blends the answers of 4 texels (free 2×2 PCF).
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
// Spot lights take one layer, point lights six (one per cube face).
@group(0) @binding(3) var local_shadows: texture_depth_2d_array;
// EnvironmentLight: the GGX-prefiltered cube (mip = roughness) and its sampler.
@group(0) @binding(4) var env_cube: texture_cube<f32>;
@group(0) @binding(5) var env_sampler: sampler;
@group(1) @binding(0) var<uniform> draw: Draw;
@group(2) @binding(0) var t_base: texture_2d<f32>;
@group(2) @binding(1) var t_normal: texture_2d<f32>;
@group(2) @binding(2) var t_metal_rough: texture_2d<f32>;
@group(2) @binding(3) var t_occlusion: texture_2d<f32>;
@group(2) @binding(4) var t_emissive: texture_2d<f32>;
@group(2) @binding(5) var s_material: sampler;

// Which texture slots are really in use (the rest are bound to a 1×1 white).
const TEX_BASE: u32 = 1u;
const TEX_NORMAL: u32 = 2u;
const TEX_METAL_ROUGH: u32 = 4u;
const TEX_OCCLUSION: u32 = 8u;
const TEX_EMISSIVE: u32 = 16u;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) texcoord: vec3f,
}

struct VsOut {
    @builtin(position) clip: vec4f,   // where on screen (the rasterizer uses this)
    @location(0) world_pos: vec3f,    // everything else is *interpolated* across the triangle
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
    @location(3) view_depth: f32,
    @location(4) uv: vec2f,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let world = draw.model * vec4f(in.position, 1.0);
    var out: VsOut;
    out.clip = frame.view_proj * world;
    out.world_pos = world.xyz;
    out.normal = (draw.normal_matrix * vec4f(in.normal, 0.0)).xyz;
    out.color = in.color;
    out.view_depth = -(frame.view * world).z;
    out.uv = (draw.texture_matrix * vec4f(in.texcoord, 1.0)).xy;
    return out;
}

// --- lines and points as screen-space quads --------------------------------
//
// 🎨 GPUs draw `LineList` and `PointList` exactly 1 pixel wide (WebGPU has
// no line width), and some drivers clip such lines badly when an end is
// behind the camera. So each segment becomes an *instance* of a 6-vertex
// quad: both ends go to clip space, the segment is cut where it crosses the
// near plane, and the quad is pushed sideways by half the width *in pixels*.
// A point is a segment whose two ends are the same: it becomes a square.

struct SegmentIn {
    @location(0) a: vec3f,
    @location(1) b: vec3f,
    @location(2) color_a: vec4f,
    @location(3) color_b: vec4f,
}

@vertex
fn vs_wide(@builtin(vertex_index) vi: u32, in: SegmentIn) -> VsOut {
    // Two triangles: (x = which end, −1 a / +1 b; y = which side).
    var corners = array(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                        vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let corner = corners[vi];
    let world_a = draw.model * vec4f(in.a, 1.0);
    let world_b = draw.model * vec4f(in.b, 1.0);
    var ca = frame.view_proj * world_a;
    var cb = frame.view_proj * world_b;
    var out: VsOut;
    // wgpu's clip space keeps 0 ≤ z ≤ w: cut what lies in front of z = 0.
    if ca.z < 0.0 && cb.z < 0.0 {
        out.clip = vec4f(0.0, 0.0, -1.0, 1.0);  // all behind us: clipped away
        return out;
    }
    if ca.z < 0.0 { ca = mix(ca, cb, ca.z / (ca.z - cb.z)); }
    if cb.z < 0.0 { cb = mix(cb, ca, cb.z / (cb.z - ca.z)); }

    let half = frame.viewport.xy * 0.5;
    let pa = ca.xy / ca.w * half;   // ends in pixels from the screen centre
    let pb = cb.xy / cb.w * half;
    let width = max(draw.line.x * frame.viewport.z, 1.0);
    let is_point = all(in.a == in.b);
    var dir = pb - pa;
    dir = select(vec2f(1.0, 0.0), normalize(dir), dot(dir, dir) > 1e-8);
    let side = vec2f(-dir.y, dir.x);
    // Points: a square. Lines: sideways by half the width, and each end
    // pushed out by half the width too (square caps, so joints close).
    var offset = (corner.x * dir + corner.y * side) * width * 0.5;
    if is_point { offset = corner * width * 0.5; }

    let at_b = corner.x > 0.0;
    let c = select(ca, cb, at_b);
    out.clip = c + vec4f(offset / half * c.w, 0.0, 0.0);
    out.world_pos = select(world_a, world_b, at_b).xyz;
    out.normal = vec3f(0.0);
    out.color = select(in.color_a, in.color_b, at_b);
    out.view_depth = -(frame.view * select(world_a, world_b, at_b)).z;
    out.uv = vec2f(0.0);
    return out;
}

// --- colour space helpers -------------------------------------------------

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    // In HDR mode (params.w bit 2) keep values above 1: the tone mapper
    // decides what to do with them. Otherwise clip, as an 8-bit screen would.
    let top = select(1.0, 1e4, (frame.params.w & 4u) != 0u);
    let x = clamp(c, vec3f(0.0), vec3f(top));
    // Bit 3: X_ITE's encode, a plain 2.2 power (x_ite.js 16.7.1 line 67921),
    // instead of the piecewise sRGB curve.
    if (frame.params.w & 8u) != 0u { return pow(x, vec3f(1.0 / 2.2)); }
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

// --- lights ---------------------------------------------------------------

// X3D light scoping: a `global='false'` light lights only its own group.
// The headlight (if on) is lights[0] and reaches everything.
fn light_reaches(i: u32) -> bool {
    let headlights = (frame.params.w >> 1u) & 1u;
    if i < headlights { return true; }
    let j = i - headlights;
    return j < 32u && ((draw.light_mask.x >> j) & 1u) == 1u;
}

// Direction towards the light, and how much of it reaches `p`
// (X3D attenuation 1 / max(c + l·d + q·d², 1), the radius, the spot cone).
struct LightSample {
    l: vec3f,
    reach: f32,
}

fn sample_light(light: Light, p: vec3f) -> LightSample {
    let kind = u32(light.position_kind.w);
    if kind == 0u {
        return LightSample(-light.direction_ambient.xyz, 1.0);
    }
    let to_light = light.position_kind.xyz - p;
    let d = length(to_light);
    let l = to_light / max(d, 1e-6);
    if d > light.attenuation_radius.w { return LightSample(l, 0.0); }
    let a = light.attenuation_radius.xyz;
    var reach = 1.0 / max(a.x + a.y * d + a.z * d * d, 1.0);
    if kind == 2u {
        let angle = acos(clamp(dot(-l, light.direction_ambient.xyz), -1.0, 1.0));
        let beam = min(light.spot.x, light.spot.y);
        let cut = light.spot.y;
        if angle >= cut {
            reach = 0.0;
        } else if angle > beam {
            reach *= (angle - cut) / (beam - cut);
        }
    }
    return LightSample(l, reach);
}

// --- shadows -----------------------------------------------------------------
//
// 🎨 A shadow map is a depth picture taken *from the light*. To shade a
// point we move it into that picture and compare: if something nearer to
// the light was recorded there, the point is in shadow.

// 3×3 PCF (each tap is already a bilinear 2×2 compare): soft-ish edges.
fn pcf(map: texture_depth_2d_array, layer: i32, uv: vec2f, depth: f32, texel: f32) -> f32 {
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let o = vec2f(f32(x), f32(y)) * texel;
            lit += textureSampleCompareLevel(map, shadow_sampler, uv + o, layer, depth);
        }
    }
    return lit / 9.0;
}

// Project `p` with shadow matrix `m`: (uv, depth, inside?).
fn to_shadow(m: mat4x4f, p: vec3f) -> vec4f {
    let q = m * vec4f(p, 1.0);
    let ndc = q.xyz / q.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0)) && ndc.z <= 1.0 && q.w > 0.0;
    return vec4f(uv, ndc.z, select(0.0, 1.0, inside));
}

// 🎨 Cascades: the sun gets 1–4 maps, each covering a farther slice of the
// view, so texels near the camera are small and far ones big. Pick the
// first cascade whose slice contains this pixel's view depth.
fn sun_shadow(p: vec3f, n: vec3f) -> f32 {
    let depth = -(frame.view * vec4f(p, 1.0)).z;
    let count = i32(frame.shadow.y);
    var c = -1;
    for (var i = 0; i < count; i++) {
        if depth <= frame.cascade_far[i] { c = i; break; }
    }
    if c < 0 { return 1.0; }
    // Nudge along the normal by ~1.5 texels: fights "shadow acne".
    let s = to_shadow(frame.shadow_mats[c], p + n * frame.cascade_texel[c] * 1.5);
    if s.w == 0.0 { return 1.0; }
    return pcf(sun_shadows, c, s.xy, s.z - frame.shadow.z, frame.shadow_maps.x);
}

// Spot: one perspective map. Point: six 90° maps; the face is the major
// axis of (p − light), in the order +X −X +Y −Y +Z −Z.
fn local_shadow(light: Light, p: vec3f, n: vec3f) -> f32 {
    var m = i32(light.spot.z);
    let to_p = p - light.position_kind.xyz;
    if u32(light.position_kind.w) == 1u {
        let a = abs(to_p);
        if a.x >= a.y && a.x >= a.z {
            m += select(1, 0, to_p.x > 0.0);
        } else if a.y >= a.z {
            m += select(3, 2, to_p.y > 0.0);
        } else {
            m += select(5, 4, to_p.z > 0.0);
        }
    }
    // The normal offset grows with distance: perspective texels do too.
    let offset = length(to_p) * frame.shadow_maps.y * 2.0;
    let s = to_shadow(frame.shadow_mats[m], p + n * offset);
    if s.w == 0.0 { return 1.0; }
    return pcf(local_shadows, m - 4, s.xy, s.z - frame.shadow.z * 0.1, frame.shadow_maps.y);
}

// How much of light `i` is unblocked at `p` (X3D shadowIntensity scales it).
fn unshadowed(i: u32, p: vec3f, n: vec3f) -> f32 {
    if i32(i) == i32(frame.shadow.x) {
        return mix(1.0, sun_shadow(p, n), frame.shadow.w);
    }
    let light = frame.lights[i];
    if light.spot.z < 0.0 { return 1.0; }
    return mix(1.0, local_shadow(light, p, n), light.spot.w);
}

// Shadow pass: only depth, seen from the light.
// Which shadow matrix this pass renders (a dynamic-offset uniform).
struct ShadowPass { index: vec4u }
@group(3) @binding(0) var<uniform> shadow_pass: ShadowPass;

@vertex
fn vs_shadow(@location(0) position: vec3f) -> @builtin(position) vec4f {
    return frame.shadow_mats[shadow_pass.index.x] * draw.model * vec4f(position, 1.0);
}

// --- X3D classic lighting (ISO/IEC 19775-1, 17.2.2.4 "Lighting equations") --
//
//   colour = emissive + Σ_lights  on · attenuation · spot · lightColour ·
//            ( ambient + diffuse + specular )
//   ambient  = light.ambientIntensity · base · material.ambientIntensity
//   diffuse  = light.intensity · base · max(N·L, 0)
//   specular = light.intensity · specularColour · max(N·H, 0)^(shininess·128)

fn shade_classic(p: vec3f, n: vec3f, base: vec3f, emissive: vec3f, specular: vec3f, linear_mode: bool) -> vec3f {
    let v = normalize(frame.camera_pos.xyz - p);
    let shininess = max(draw.specular.w * 128.0, 1.0);
    var sum = vec3f(0.0);
    for (var i = 0u; i < frame.params.x; i++) {
        if !light_reaches(i) { continue; }
        let light = frame.lights[i];
        let s = sample_light(light, p);
        var color = light.color_intensity.rgb;
        if linear_mode { color = srgb_to_linear(color); }
        let intensity = light.color_intensity.w;
        let ambient = light.direction_ambient.w * base * draw.emissive.w;
        let n_dot_l = max(dot(n, s.l), 0.0);
        let diffuse = intensity * base * n_dot_l;
        var spec = vec3f(0.0);
        if n_dot_l > 0.0 {
            let h = normalize(s.l + v);  // Blinn's half vector
            spec = intensity * specular * pow(max(dot(n, h), 0.0), shininess);
        }
        sum += s.reach * color * (ambient + unshadowed(i, p, n) * (diffuse + spec));
    }
    return clamp(emissive + sum, vec3f(0.0), vec3f(1.0));
}

// --- PhysicalMaterial: metallic-roughness PBR, in linear space ---------------
//
// The glTF BRDF: Lambert diffuse (base/π) + GGX microfacets (D), the
// height-correlated Smith visibility (V) and Schlick's Fresnel (F).
// A light of intensity i gives irradiance i·N·L — that is what X_ITE does
// (measured: baseColor 0.8 under a DirectionalLight of 0.3 → 78/255).

fn shade_pbr(p: vec3f, n: vec3f, base: vec3f, metallic: f32, roughness: f32, ao: f32, emissive: vec3f) -> vec3f {
    let v = normalize(frame.camera_pos.xyz - p);
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3f(0.04), base, metallic);
    let diffuse_color = base * (1.0 - metallic);
    let a = max(roughness * roughness, 0.002);
    let a2 = a * a;
    var sum = vec3f(0.0);
    for (var i = 0u; i < frame.params.x; i++) {
        if !light_reaches(i) { continue; }
        let light = frame.lights[i];
        let s = sample_light(light, p);
        let radiance = light.color_intensity.rgb * light.color_intensity.w * s.reach * unshadowed(i, p, n);
        sum += light.direction_ambient.w * light.color_intensity.rgb * diffuse_color * ao * s.reach;
        let n_dot_l = max(dot(n, s.l), 0.0);
        if n_dot_l <= 0.0 { continue; }
        let h = normalize(s.l + v);
        let n_dot_h = max(dot(n, h), 0.0);
        let v_dot_h = max(dot(v, h), 0.0);
        let dd = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * dd * dd);
        let vis = 0.5 / (n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2)
                       + n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2));
        let f = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
        sum += ((1.0 - f) * diffuse_color / PI + d * vis * f) * radiance * n_dot_l;
    }
    if frame.env.x > 0.0 {
        sum += environment(n, v, n_dot_v, f0, diffuse_color, roughness, ao);
    }
    return emissive + sum;
}

// --- EnvironmentLight (image-based lighting, see environment.rs) ------------

fn sh_irradiance(n: vec3f) -> vec3f {
    let c = frame.env_sh;
    return c[0].rgb * 0.282095
        + c[1].rgb * (0.488603 * n.y) + c[2].rgb * (0.488603 * n.z) + c[3].rgb * (0.488603 * n.x)
        + c[4].rgb * (1.092548 * n.x * n.y) + c[5].rgb * (1.092548 * n.y * n.z)
        + c[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0)) + c[7].rgb * (1.092548 * n.x * n.z)
        + c[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
}

// Karis's analytic fit of the split-sum DFG term ("Physically Based Shading
// on Mobile", 2014): (scale, bias) so that specular = F0 · scale + bias.
fn env_brdf(roughness: f32, n_dot_v: f32) -> vec2f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2f(-1.04, 1.04) * a004 + r.zw;
}

fn environment(n: vec3f, v: vec3f, n_dot_v: f32, f0: vec3f, diffuse_color: vec3f, roughness: f32, ao: f32) -> vec3f {
    let to_env = mat3x3f(frame.env_rotation[0].xyz, frame.env_rotation[1].xyz, frame.env_rotation[2].xyz);
    let irradiance = max(sh_irradiance(to_env * n), vec3f(0.0));
    let r = to_env * reflect(-v, n);
    let lod = roughness * (frame.env.y - 1.0);
    let prefiltered = textureSampleLevel(env_cube, env_sampler, r, lod).rgb;
    let ab = env_brdf(roughness, n_dot_v);
    let fss_ess = f0 * ab.x + ab.y; // single-scattering specular
    let e = irradiance / PI;        // cosine-weighted mean radiance
    let model = u32(frame.env.z);
    if model == 2u {
        // 🎨 Multiple scattering (Fdez-Agüera 2019): light a single GGX bounce
        // loses (1 − Ess) reappears, tinted by the average Fresnel. A white
        // metal in a white furnace then stays white at every roughness.
        let ess = ab.x + ab.y;
        let ems = 1.0 - ess;
        let f_avg = f0 + (vec3f(1.0) - f0) / 21.0;
        let fms_ems = fss_ess * f_avg / (vec3f(1.0) - ems * f_avg) * ems;
        let kd = diffuse_color * (vec3f(1.0) - fss_ess - fms_ems);
        return frame.env_color.rgb * ao * (fss_ess * prefiltered + (fms_ems + kd) * e);
    }
    // 1: energy split (diffuse gets what specular didn't); 0: plain sum.
    let kd = select(vec3f(1.0), vec3f(1.0) - fss_ess, model == 1u);
    return frame.env_color.rgb * ao * (diffuse_color * kd * e + prefiltered * fss_ess);
}

// 🎨 Normal mapping *without* precomputed tangents: rebuild the tangent
// frame per pixel from how position and uv change across the screen
// (screen-space derivatives). Christian Schüler, "Normal Mapping Without
// Precomputed Tangents" (2013).
fn perturb_normal(n: vec3f, dp_dx: vec3f, dp_dy: vec3f, duv_dx: vec2f, duv_dy: vec2f, sampled: vec3f, scale: f32) -> vec3f {
    let dp2perp = cross(dp_dy, n);
    let dp1perp = cross(n, dp_dx);
    let t = dp2perp * duv_dx.x + dp1perp * duv_dy.x;
    let b = dp2perp * duv_dx.y + dp1perp * duv_dy.y;
    let len = max(dot(t, t), dot(b, b));
    if len < 1e-20 { return n; }  // degenerate uv: keep the geometric normal
    let inv = inverseSqrt(len);
    let m = (sampled * 2.0 - 1.0) * vec3f(scale, scale, 1.0);
    return normalize(mat3x3f(t * inv, b * inv, n) * m);
}

// Debug views (keys 1–7 in the viewer, `--view` on the command line).
const VIEW_LIT: u32 = 0u;
const VIEW_NORMALS: u32 = 1u;
const VIEW_DEPTH: u32 = 2u;
const VIEW_BASE_COLOR: u32 = 3u;
const VIEW_LIGHTING_ONLY: u32 = 4u;
const VIEW_FACING: u32 = 5u;
const VIEW_UV: u32 = 6u;

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    // Texture sampling and derivatives must happen in *uniform control flow*
    // (every pixel of a 2×2 quad together), so do them all first, before
    // any branch that depends on per-pixel values.
    let uv = vec2f(in.uv.x, 1.0 - in.uv.y);  // X3D's t runs up the image; texel rows run down
    let tex_base = textureSample(t_base, s_material, uv);
    let tex_normal = textureSample(t_normal, s_material, uv).rgb;
    let tex_mr = textureSample(t_metal_rough, s_material, uv);
    let tex_occlusion = textureSample(t_occlusion, s_material, uv).r;
    let tex_emissive = textureSample(t_emissive, s_material, uv).rgb;
    let dp_dx = dpdx(in.world_pos);
    let dp_dy = dpdy(in.world_pos);
    let duv_dx = dpdx(in.uv);
    let duv_dy = dpdy(in.uv);

    let linear_mode = frame.params.z == 1u;
    let unlit = draw.flags.x == 1u;
    let vertex_colors = draw.flags.y == 1u;
    let physical = draw.flags.z == 1u;
    let bits = draw.flags.w;

    // Two-sided lighting: a back face uses the flipped normal.
    var n = in.normal;
    if dot(n, n) < 1e-12 { n = vec3f(0.0, 0.0, 1.0); }  // lines and points have none
    n = normalize(n);
    if !front { n = -n; }
    if (bits & TEX_NORMAL) != 0u && !unlit {
        let scale = select(1.0, draw.pbr.w, physical);
        n = perturb_normal(n, dp_dx, dp_dy, duv_dx, duv_dy, tex_normal, scale);
    }

    var alpha = draw.diffuse.a;
    var rgb: vec3f;
    var base_shown: vec3f;
    if physical && !unlit {
        // PhysicalMaterial colours are linear (X3D 4: LINEAR_WHEN_PHYSICAL_MATERIAL);
        // its textures are ordinary sRGB images.
        var base = draw.diffuse.rgb;
        if vertex_colors { base *= in.color.rgb; alpha *= in.color.a; }
        if (bits & TEX_BASE) != 0u { base *= srgb_to_linear(tex_base.rgb); alpha *= tex_base.a; }
        var metallic = draw.pbr.x;
        var roughness = draw.pbr.y;
        if (bits & TEX_METAL_ROUGH) != 0u { roughness *= tex_mr.g; metallic *= tex_mr.b; }
        var ao = 1.0;
        if (bits & TEX_OCCLUSION) != 0u { ao = mix(1.0, tex_occlusion, draw.pbr.z); }
        var emissive = draw.emissive.rgb;
        if (bits & TEX_EMISSIVE) != 0u { emissive *= srgb_to_linear(tex_emissive); }
        let exposure = frame.camera_pos.w;
        rgb = linear_to_srgb(shade_pbr(in.world_pos, n, base, metallic, roughness, ao, emissive) * exposure);
        base_shown = linear_to_srgb(base);
    } else {
        // X3D: a Color node *replaces* the diffuse colour (or the emissive one when unlit).
        var base = draw.diffuse.rgb;
        var emissive = draw.emissive.rgb;
        if vertex_colors {
            base = in.color.rgb;
            if unlit { emissive = in.color.rgb; }
            alpha *= in.color.a;
        }
        if (bits & TEX_BASE) != 0u {
            base *= tex_base.rgb;
            if unlit { emissive *= tex_base.rgb; }
            alpha *= tex_base.a;
        }
        if (bits & TEX_EMISSIVE) != 0u { emissive *= tex_emissive; }
        var specular = draw.specular.rgb;
        if unlit {
            rgb = emissive;
        } else if linear_mode {
            rgb = linear_to_srgb(shade_classic(in.world_pos, n, srgb_to_linear(base), srgb_to_linear(emissive),
                                               srgb_to_linear(specular), true));
        } else {
            rgb = shade_classic(in.world_pos, n, base, emissive, specular, false);
        }
        base_shown = select(base, emissive, unlit);
    }

    // Fog (19775-1 24.2.1): toward the fog colour with the distance from the eye.
    if frame.fog.x > 0.5 {
        let d = length(in.world_pos - frame.camera_pos.xyz);
        var f = clamp((frame.fog.y - d) / frame.fog.y, 0.0, 1.0);
        if frame.fog.x > 1.5 { f = select(exp(-d / (frame.fog.y - d)), 0.0, d >= frame.fog.y); }
        rgb = mix(frame.fog_color.rgb, rgb, f);
    }

    switch frame.params.y {
        case VIEW_NORMALS: { rgb = n * 0.5 + 0.5; }
        case VIEW_DEPTH: {
            let t = (in.view_depth - frame.near_far.z) / (frame.near_far.w - frame.near_far.z);
            rgb = vec3f(1.0 - clamp(t, 0.0, 1.0));
        }
        case VIEW_BASE_COLOR: { rgb = base_shown; }
        case VIEW_LIGHTING_ONLY: {
            rgb = select(shade_classic(in.world_pos, n, vec3f(0.8), vec3f(0.0), vec3f(0.0), false), vec3f(1.0), unlit);
        }
        case VIEW_FACING: { rgb = select(vec3f(0.9, 0.2, 0.2), vec3f(0.2, 0.8, 0.3), front); }
        case VIEW_UV: { rgb = vec3f(fract(in.uv), 0.0); }
        default: {}
    }

    // An sRGB target encodes on write, so hand it linear values.
    if (frame.params.w & 1u) != 0u { rgb = srgb_to_linear(rgb); }
    return vec4f(rgb, alpha);
}
