// Post-processing: bloom and tone mapping, after the scene is drawn.
//
// The scene pass wrote *display* values (what X_ITE would show) into a
// 16-bit float texture, so nothing above 1.0 was clipped. Here we:
//
//   1. prefilter  full-res HDR ─▶ bloom mip 0 (half res): keep only the
//                 bright part (soft threshold), in linear light
//   2. down       mip i-1 ─▶ mip i   13-tap filter (Jimenez, COD:AW 2014)
//   3. up         mip i+1 ─▶ mip i   3×3 tent, *added* to what is there
//   4. composite  HDR + strength × mip 0 → exposure → tone map → display
//
// Every pass draws one full-screen triangle; the fragment shader does the work.

struct Post {
    // x: bloom strength, y: threshold, z: soft knee, w: exposure
    bloom: vec4f,
    // x: tone map (0 off, 1 ACES, 2 AgX), y: 1 = target is sRGB (encodes on write)
    mode: vec4u,
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> post: Post;
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

// 🎨 One triangle that covers the screen: (-1,-1), (3,-1), (-1,3).
// No vertex buffer needed — the index is all we use.
@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> VsOut {
    let xy = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.position = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2f(xy.x, 1.0 - xy.y);
    return out;
}

// sRGB decode/encode, extended past 1.0 so bright values survive.
fn to_linear(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045));
}

fn to_display(c: vec3f) -> vec3f {
    let x = clamp(c, vec3f(0.0), vec3f(1.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3f(0.0031308));
}

fn tap(t: texture_2d<f32>, uv: vec2f) -> vec3f {
    return textureSampleLevel(t, smp, uv, 0.0).rgb;
}

// 13 taps in a 4×4 texel footprint, weighted so bright single pixels
// don't flicker as the camera moves.
fn downsample13(uv: vec2f) -> vec3f {
    let d = 1.0 / vec2f(textureDimensions(source));
    let a = tap(source, uv + d * vec2f(-2.0, -2.0));
    let b = tap(source, uv + d * vec2f(0.0, -2.0));
    let c = tap(source, uv + d * vec2f(2.0, -2.0));
    let e = tap(source, uv + d * vec2f(-2.0, 0.0));
    let f = tap(source, uv);
    let g = tap(source, uv + d * vec2f(2.0, 0.0));
    let h = tap(source, uv + d * vec2f(-2.0, 2.0));
    let i = tap(source, uv + d * vec2f(0.0, 2.0));
    let j = tap(source, uv + d * vec2f(2.0, 2.0));
    let k = tap(source, uv + d * vec2f(-1.0, -1.0));
    let l = tap(source, uv + d * vec2f(1.0, -1.0));
    let m = tap(source, uv + d * vec2f(-1.0, 1.0));
    let n = tap(source, uv + d * vec2f(1.0, 1.0));
    return f * 0.125 + (a + c + h + j) * 0.03125 + (b + e + g + i) * 0.0625 + (k + l + m + n) * 0.125;
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4f {
    // The scene texture holds display values: decode them first.
    let d = 1.0 / vec2f(textureDimensions(source));
    var c = vec3f(0.0);
    for (var y = -1.0; y <= 1.0; y += 2.0) {
        for (var x = -1.0; x <= 1.0; x += 2.0) {
            c += to_linear(tap(source, in.uv + d * vec2f(x, y)));
        }
    }
    c = c * 0.25 * post.bloom.w;
    // Soft threshold: a quadratic ramp of width 2·knee around the threshold.
    let brightness = max(c.r, max(c.g, c.b));
    let t = post.bloom.y;
    let knee = max(t * post.bloom.z, 1e-4);
    let soft = clamp(brightness - t + knee, 0.0, 2.0 * knee);
    let contribution = max(soft * soft / (4.0 * knee), brightness - t) / max(brightness, 1e-4);
    return vec4f(c * max(contribution, 0.0), 1.0);
}

@fragment
fn fs_down(in: VsOut) -> @location(0) vec4f {
    return vec4f(downsample13(in.uv), 1.0);
}

@fragment
fn fs_up(in: VsOut) -> @location(0) vec4f {
    // 3×3 tent; the pipeline blends One + One onto the larger mip.
    let d = 1.0 / vec2f(textureDimensions(source));
    var c = tap(source, in.uv) * 4.0;
    c += (tap(source, in.uv + vec2f(-d.x, 0.0)) + tap(source, in.uv + vec2f(d.x, 0.0))
        + tap(source, in.uv + vec2f(0.0, -d.y)) + tap(source, in.uv + vec2f(0.0, d.y))) * 2.0;
    c += tap(source, in.uv - d) + tap(source, in.uv + d)
        + tap(source, in.uv + vec2f(d.x, -d.y)) + tap(source, in.uv + vec2f(-d.x, d.y));
    return vec4f(c / 16.0, 1.0);
}

// --- tone mapping ----------------------------------------------------------

// ACES filmic, Krzysztof Narkowicz's fit (2015): one rational curve.
fn aces(x: vec3f) -> vec3f {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
}

// AgX (Troy Sobotka), via Benjamin Wrensch's polynomial fit: desaturates
// bright colours toward white instead of skewing hue the way ACES does.
fn agx_contrast(x: vec3f) -> vec3f {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3f) -> vec3f {
    // 🎨 WGSL matrices are column-major: each vec3 below is a *column*.
    let inset = mat3x3f(
        vec3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3f(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3f(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3f(
        vec3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3f(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3f(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = inset * max(c, vec3f(1e-10));
    x = clamp(log2(x), vec3f(min_ev), vec3f(max_ev));
    x = (x - min_ev) / (max_ev - min_ev);
    x = agx_contrast(x);
    x = outset * x;
    // AgX's curve lands in display space; decode back to linear.
    return to_linear(clamp(x, vec3f(0.0), vec3f(1.0)));
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4f {
    let scene = textureSampleLevel(source, smp, in.uv, 0.0);
    var c = to_linear(scene.rgb) * post.bloom.w;
    c += tap(bloom_tex, in.uv) * post.bloom.x;
    switch post.mode.x {
        case 1u: { c = aces(c); }
        case 2u: { c = agx(c); }
        default: {}
    }
    // An sRGB target encodes on write; anything else gets display values.
    let out = select(to_display(c), clamp(c, vec3f(0.0), vec3f(1.0)), post.mode.y == 1u);
    return vec4f(out, 1.0);
}
