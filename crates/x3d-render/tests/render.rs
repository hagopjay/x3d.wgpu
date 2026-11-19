//! GPU integration tests: render tiny scenes and check actual pixels.
//!
//! They need *some* GPU adapter. On a machine without one (CI, containers),
//! install a software Vulkan driver (`apt install mesa-vulkan-drivers`); if
//! no adapter exists at all the GPU tests print SKIPPED and pass, unless
//! `X3D_REQUIRE_GPU=1` is set, which makes that a failure (CI sets it).

use image::{Rgba, RgbaImage};
use x3d_render::renderer::BUILTIN_SHADER;
use x3d_render::uniforms::{DrawUniform, FrameUniform};
use x3d_render::{ColorMode, Gpu, RenderSettings, Renderer, Shot, render_scene};

/// The adapter for a GPU test. With `X3D_REQUIRE_GPU` set (CI does), no
/// adapter is a *failure*, so the GPU tests can't pass by skipping.
fn gpu() -> Option<Gpu> {
    static NAMED: std::sync::Once = std::sync::Once::new();
    match Gpu::headless(x3d_render::Backend::Auto) {
        Ok(g) => {
            NAMED.call_once(|| eprintln!("GPU tests on: {}", g.describe()));
            Some(g)
        }
        Err(e) if std::env::var_os("X3D_REQUIRE_GPU").is_some() => {
            panic!("X3D_REQUIRE_GPU is set but there is no adapter: {e}")
        }
        Err(e) => {
            eprintln!("SKIPPED GPU test (no adapter): {e}");
            None
        }
    }
}

fn scene(body: &str) -> x3d_scene::Scene {
    x3d_scene::parse_str(&format!("<X3D><Scene><Viewpoint position='0 0 5'/>{body}</Scene></X3D>")).unwrap()
}

fn shot(settings: RenderSettings) -> Shot {
    Shot { size: (64, 64), msaa: 1, settings, viewpoint: None }
}

fn center(img: &RgbaImage) -> [u8; 4] {
    img.get_pixel(img.width() / 2, img.height() / 2).0
}

fn close(a: [u8; 4], b: [u8; 4], tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

#[test]
fn lit_box_on_a_blue_background() {
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Background skyColor='0 0 1'/>
         <Shape><Appearance><Material diffuseColor='1 0 0'/></Appearance><Box size='1 1 1'/></Shape>",
    );
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    let c = center(&img);
    // Headlight straight on: diffuse 1 × N·L 1 = full red.
    assert!(close(c, [255, 0, 0, 255], 2), "centre {c:?}");
    assert_eq!(img.get_pixel(0, 0), &Rgba([0, 0, 255, 255]), "corner shows the Background");
}

#[test]
fn unlit_colours_reach_the_screen_exactly() {
    let Some(gpu) = gpu() else { return };
    let s = scene("<Shape><Appearance><UnlitMaterial emissiveColor='0.5 0.25 1'/></Appearance><Box/></Shape>");
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert!(close(center(&img), [128, 64, 255, 255], 1), "{:?}", center(&img));
}

#[test]
fn linear_mode_lights_differently() {
    let Some(gpu) = gpu() else { return };
    // A 45° light: N·L ≈ 0.707, so display and linear lighting disagree.
    let s = scene(
        "<NavigationInfo headlight='false'/>
         <DirectionalLight direction='0 -0.7071 -0.7071'/>
         <Shape><Appearance><Material diffuseColor='0.8 0.8 0.8'/></Appearance><Box size='3 3 0.1'/></Shape>",
    );
    let display = center(&render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap());
    let linear_settings = RenderSettings { color_mode: ColorMode::Linear, ..Default::default() };
    let linear = center(&render_scene(&gpu, &s, &shot(linear_settings)).unwrap());
    // display: 0.8 × 0.707 = 0.566 → 144. linear: srgb(lin(0.8) × 0.707) ≈ 0.69 → 175.
    assert!(close(display, [144, 144, 144, 255], 2), "display {display:?}");
    assert!(close(linear, [175, 175, 175, 255], 3), "linear {linear:?}");
}

#[test]
fn transparent_quad_blends_over_an_opaque_one() {
    let Some(gpu) = gpu() else { return };
    // Listed front-to-back on purpose: the renderer must still draw the
    // transparent one last.
    let s = scene(
        "<Background skyColor='0 0 0'/>
         <Shape><Appearance><UnlitMaterial emissiveColor='1 1 1' transparency='0.5'/></Appearance><Box size='2 2 0.01'/></Shape>
         <Transform translation='0 0 -1'>
           <Shape><Appearance><UnlitMaterial emissiveColor='1 0 0'/></Appearance><Box size='4 4 0.01'/></Shape>
         </Transform>",
    );
    let c = center(&render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap());
    assert!(close(c, [255, 128, 128, 255], 2), "50 % white over red should be pink: {c:?}");
}

#[test]
fn a_broken_shader_is_rejected_and_the_old_one_keeps_working() {
    let Some(gpu) = gpu() else { return };
    let s = scene("<Shape><Appearance><UnlitMaterial emissiveColor='0 1 0'/></Appearance><Box/></Shape>");
    let mut renderer = Renderer::new(&gpu, wgpu::TextureFormat::Rgba8Unorm, 1);
    renderer.set_scene(&s);

    let err = renderer.reload_shader("fn this is not wgsl").unwrap_err();
    assert!(!err.is_empty());

    // A shader that parses but whose bindings don't match the Rust side.
    let mismatched = BUILTIN_SHADER.replace("@group(1) @binding(0)", "@group(1) @binding(7)");
    assert!(renderer.reload_shader(&mismatched).is_err());

    let img = render_with(&gpu, &mut renderer, &s);
    assert!(close(center(&img), [0, 255, 0, 255], 1), "old shader still draws");

    // And a good edit takes effect.
    let magenta = BUILTIN_SHADER.replace("return vec4f(rgb, alpha);", "return vec4f(1.0, 0.0, 1.0, 1.0);");
    renderer.reload_shader(&magenta).unwrap();
    let img = render_with(&gpu, &mut renderer, &s);
    assert!(close(center(&img), [255, 0, 255, 255], 1));
}

fn render_with(gpu: &Gpu, renderer: &mut Renderer, s: &x3d_scene::Scene) -> RgbaImage {
    let target = x3d_render::headless::Offscreen::new(&gpu.device, (32, 32), renderer.color_format());
    let camera = x3d_render::headless::camera_for(s, None);
    renderer.render(&target.view, (32, 32), &camera, &RenderSettings::default());
    target.read(gpu).unwrap()
}

#[test]
fn odd_widths_read_back_correctly() {
    // 37 px × 4 bytes = 148 bytes per row, padded to 256 on the GPU side.
    let Some(gpu) = gpu() else { return };
    let s = scene("<Background skyColor='1 0.5 0'/>");
    let img = render_scene(&gpu, &s, &Shot { size: (37, 5), ..shot(RenderSettings::default()) }).unwrap();
    assert_eq!(img.dimensions(), (37, 5));
    assert!(img.pixels().all(|p| close(p.0, [255, 128, 0, 255], 1)));
}

#[test]
fn contact_sheet_is_four_by_two() {
    let Some(gpu) = gpu() else { return };
    let s = scene("<Shape><Box/></Shape>");
    let sheet = x3d_render::render_sheet(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert_eq!(sheet.dimensions(), (64 * 4, 64 * 2));
}

/// No GPU needed: ask naga (wgpu's shader compiler) how big the WGSL structs
/// are, and compare with the Rust mirrors. Catches layout drift instantly.
#[test]
fn rust_uniforms_match_the_wgsl_layout() {
    let module = naga::front::wgsl::parse_str(BUILTIN_SHADER).expect("scene.wgsl parses");
    let size_of_wgsl = |name: &str| {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { span, .. } if ty.name.as_deref() == Some(name) => Some(*span as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("struct {name} not in the shader"))
    };
    assert_eq!(size_of_wgsl("Frame"), size_of::<FrameUniform>());
    assert_eq!(size_of_wgsl("Draw"), size_of::<DrawUniform>());
    assert_eq!(size_of_wgsl("Light"), size_of::<x3d_render::uniforms::GpuLight>());

    // Full validation, as wgpu would do at runtime.
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .expect("scene.wgsl validates");
}

#[test]
fn physical_material_matches_x_ite() {
    // x3d.lab measured X_ITE 16.7: PhysicalMaterial baseColor 0.8 under a
    // head-on DirectionalLight of 0.3 draws as 78/255.
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<NavigationInfo headlight='false'/>
         <DirectionalLight direction='0 0 -1' intensity='0.3'/>
         <Shape><Appearance><PhysicalMaterial baseColor='0.8 0.8 0.8' metallic='0' roughness='1'/></Appearance>
           <Box size='3 3 0.1'/></Shape>",
    );
    let c = center(&render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap());
    assert!(close(c, [78, 78, 78, 255], 2), "PBR centre {c:?}, X_ITE gives 78");
}

#[test]
fn textures_are_sampled_with_x3d_orientation() {
    // A 2×2 texture: top row red|green, bottom row blue|white. X3D's t runs
    // *up*, so (s, t) = (0.25, 0.75) is the top-left texel: red.
    let Some(gpu) = gpu() else { return };
    let dir = std::env::temp_dir().join(format!("x3d-render-tex-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut img = RgbaImage::new(2, 2);
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
    img.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
    img.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
    img.save(dir.join("quad.png")).unwrap();
    // A 1×1 quad filling the view, unlit, with the texture moved so the
    // centre pixel samples the top-left texel's centre.
    std::fs::write(
        dir.join("t.x3d"),
        "<X3D><Scene><Viewpoint position='0 0 1' fieldOfView='0.2'/>
           <Shape><Appearance><UnlitMaterial/><ImageTexture url='\"quad.png\"'/>
             <TextureTransform translation='-0.25 0.25'/></Appearance>
             <IndexedFaceSet coordIndex='0 1 2 3'>
               <Coordinate point='-1 -1 0  1 -1 0  1 1 0  -1 1 0'/>
               <TextureCoordinate point='0 0  1 0  1 1  0 1'/>
             </IndexedFaceSet></Shape></Scene></X3D>",
    )
    .unwrap();
    let s = x3d_scene::load(dir.join("t.x3d")).unwrap();
    let c = center(&render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap());
    std::fs::remove_dir_all(&dir).ok();
    assert!(close(c, [255, 0, 0, 255], 8), "expected the top-left (red) texel, got {c:?}");
}

/// Count the pixels in column x that aren't the background.
fn painted_rows(img: &RgbaImage, x: u32, background: [u8; 4]) -> u32 {
    (0..img.height()).filter(|&y| !close(img.get_pixel(x, y).0, background, 8)).count() as u32
}

#[test]
fn lines_have_their_x3d_width() {
    // LineProperties linewidthScaleFactor='5': a horizontal line 5 px tall.
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Shape><Appearance><UnlitMaterial emissiveColor='1 1 1'/><LineProperties linewidthScaleFactor='5'/></Appearance>
           <IndexedLineSet coordIndex='0 1'><Coordinate point='-3 0 0  3 0 0'/></IndexedLineSet></Shape>",
    );
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert_eq!(painted_rows(&img, 32, [0, 0, 0, 255]), 5);
}

#[test]
fn a_line_running_behind_the_camera_is_clipped_not_smeared() {
    // From in front of the camera (z = 0) to far behind it (z = 50). Before
    // the near-plane clip in `vs_wide`, some drivers filled whole tiles.
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Shape><Appearance><UnlitMaterial emissiveColor='1 1 1'/></Appearance>
           <IndexedLineSet coordIndex='0 1'><Coordinate point='0.5 -0.5 0  0.5 -0.5 50'/></IndexedLineSet></Shape>",
    );
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    let painted = img.pixels().filter(|p| p.0[0] > 8).count();
    assert!(painted > 0 && painted < 64 * 3, "a thin line, not a filled block: {painted} px");
}

/// A tiny ComposedShader in the GLSL ES 300 dialect X_ITE accepts.
fn composed(tint: &str, geometry: &str) -> String {
    format!(
        "<Shape><Appearance>
           <Material/>
           <ComposedShader DEF='Paint' language='GLSL'>
             <field name='tint' type='SFVec3f' accessType='inputOutput' value='{tint}'/>
             <ShaderPart type='VERTEX'><![CDATA[#version 300 es
               precision highp float;
               uniform mat4 x3d_ProjectionMatrix;
               uniform mat4 x3d_ModelViewMatrix;
               in vec4 x3d_Vertex;
               void main() {{ gl_Position = x3d_ProjectionMatrix * x3d_ModelViewMatrix * x3d_Vertex; }}
             ]]></ShaderPart>
             <ShaderPart type='FRAGMENT'><![CDATA[#version 300 es
               precision highp float;
               uniform vec3 tint;
               out vec4 fragColor;
               void main() {{ fragColor = vec4(tint, 1.0); }}
             ]]></ShaderPart>
           </ComposedShader>
         </Appearance>{geometry}</Shape>"
    )
}

#[test]
fn a_composed_shader_paints_with_its_own_uniform_field() {
    let Some(gpu) = gpu() else { return };
    let s = scene(&composed("0 1 0", "<Box size='2 2 2'/>"));
    assert_eq!(s.shaders.len(), 1, "one ComposedShader parsed");
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert!(close(center(&img), [0, 255, 0, 255], 1), "{:?}", center(&img));
}

#[test]
fn a_route_into_a_shader_field_changes_the_picture() {
    let Some(gpu) = gpu() else { return };
    let mut s = scene(&format!(
        "{}
         <TimeSensor DEF='Clock' cycleInterval='1' loop='true'/>
         <ColorInterpolator DEF='Fade' key='0 1' keyValue='1 0 0, 1 0 0'/>
         <ROUTE fromNode='Clock' fromField='fraction_changed' toNode='Fade' toField='set_fraction'/>
         <ROUTE fromNode='Fade' fromField='value_changed' toNode='Paint' toField='tint'/>",
        composed("0 1 0", "<Box size='2 2 2'/>")
    ));
    s.advance(0.5);
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert!(close(center(&img), [255, 0, 0, 255], 1), "the route made it red: {:?}", center(&img));
}

#[test]
fn an_rgba_colour_node_makes_a_shape_blend() {
    let Some(gpu) = gpu() else { return };
    // Same half-transparent red quad twice; only the ColorRGBA one is blended
    // (X_ITE's rule): the plain Color one ignores the alpha and stays opaque.
    let quad = |node: &str| {
        format!(
            "<Shape><Appearance><UnlitMaterial/></Appearance>
               <IndexedFaceSet coordIndex='0 1 2 3 -1' solid='false'>
                 <Coordinate point='-2 -2 0, 2 -2 0, 2 2 0, -2 2 0'/>{node}
               </IndexedFaceSet></Shape>"
        )
    };
    let bg = "<Background skyColor='0 0 1'/>";
    let rgba = scene(&format!("{bg}{}", quad("<ColorRGBA color='1 0 0 0.5, 1 0 0 0.5, 1 0 0 0.5, 1 0 0 0.5'/>")));
    let rgb = scene(&format!("{bg}{}", quad("<Color color='1 0 0, 1 0 0, 1 0 0, 1 0 0'/>")));
    let blended = center(&render_scene(&gpu, &rgba, &shot(RenderSettings::default())).unwrap());
    let opaque = center(&render_scene(&gpu, &rgb, &shot(RenderSettings::default())).unwrap());
    assert!(close(blended, [128, 0, 128, 255], 3), "blended {blended:?}");
    assert!(close(opaque, [255, 0, 0, 255], 1), "opaque {opaque:?}");
}

#[test]
fn aces_tone_mapping_follows_its_curve() {
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Shape><Appearance><UnlitMaterial emissiveColor='0.5 0.5 0.5'/></Appearance><Box size='3 3 3'/></Shape>",
    );
    let settings = RenderSettings { tone_map: x3d_render::ToneMap::Aces, ..Default::default() };
    let c = center(&render_scene(&gpu, &s, &shot(settings)).unwrap());
    // display 0.5 → linear 0.214 → ACES 0.3215 → display 0.603 → 154.
    assert!(close(c, [154, 154, 154, 255], 3), "{c:?}");
}

#[test]
fn bloom_spills_light_past_a_bright_edge() {
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Background skyColor='0 0 0'/>
         <Shape><Appearance><UnlitMaterial emissiveColor='1 1 1'/></Appearance><Box size='0.6 0.6 0.6'/></Shape>",
    );
    let render = |bloom: f32| {
        let settings = RenderSettings { bloom, ..Default::default() };
        // MSAA on: the HDR path must resolve into its float texture too.
        render_scene(&gpu, &s, &Shot { size: (64, 64), msaa: 4, settings, viewpoint: None }).unwrap()
    };
    let (plain, glowing) = (render(0.0), render(0.5));
    let beside = |img: &RgbaImage| img.get_pixel(32, 12).0; // above the box, on the black background
    assert_eq!(beside(&plain), [0, 0, 0, 255], "no bloom: black sky");
    assert!(beside(&glowing)[0] > 10, "bloom: the box glows onto the sky {:?}", beside(&glowing));
    assert!(close(center(&glowing), [255, 255, 255, 255], 1), "the box itself stays white");
}

#[test]
fn the_sun_casts_a_shadow_when_asked() {
    let Some(gpu) = gpu() else { return };
    // A floor, a slab hovering over it, and a sun at 45° shining toward +x,
    // so the slab's shadow lands one unit to its right. Camera: straight down.
    let s = scene(
        "<NavigationInfo headlight='false'/>
         <Viewpoint position='0 6 0' orientation='1 0 0 -1.5708'/>
         <DirectionalLight direction='1 -1 0' ambientIntensity='0'/>
         <Shape><Appearance><Material diffuseColor='1 1 1'/></Appearance>
           <IndexedFaceSet coordIndex='0 1 2 3 -1' solid='false'>
             <Coordinate point='-3 0 -3, -3 0 3, 3 0 3, 3 0 -3'/></IndexedFaceSet></Shape>
         <Transform translation='0 1 0'><Shape><Appearance><Material diffuseColor='1 1 1'/></Appearance>
           <Box size='1 0.2 1'/></Shape></Transform>",
    );
    let render = |shadows: bool| {
        let settings = RenderSettings { shadows, ..Default::default() };
        render_scene(&gpu, &s, &Shot { size: (64, 64), msaa: 1, settings, viewpoint: Some(1) }).unwrap()
    };
    let (plain, shadowed) = (render(false), render(true));
    // ~13 pixels per unit: x = −1.5 is open floor, x = +1 is in the shadow.
    let (open, shade) = (|img: &RgbaImage| img.get_pixel(12, 32).0[0], |img: &RgbaImage| img.get_pixel(45, 32).0[0]);
    assert!(open(&plain) > 150, "the floor is lit: {}", open(&plain));
    assert_eq!(open(&plain), open(&shadowed), "shadows leave open floor alone");
    assert_eq!(shade(&plain), open(&plain), "without shadows the whole floor is lit");
    assert!(shade(&shadowed) < 30, "the slab's shadow: {}", shade(&shadowed));
}

/// A floor, a slab over it, a camera straight down; `light` is spliced in.
fn shadow_stage(light: &str) -> x3d_scene::Scene {
    scene(&format!(
        "<NavigationInfo headlight='false'/>
         <Viewpoint position='0 6 0' orientation='1 0 0 -1.5708'/>
         {light}
         <Shape><Appearance><Material diffuseColor='1 1 1'/></Appearance>
           <IndexedFaceSet coordIndex='0 1 2 3 -1' solid='false'>
             <Coordinate point='-3 0 -3, -3 0 3, 3 0 3, 3 0 -3'/></IndexedFaceSet></Shape>
         <Transform translation='0 1 0'><Shape><Appearance><Material diffuseColor='1 1 1'/></Appearance>
           <Box size='1 0.2 1'/></Shape></Transform>"
    ))
}

fn shadow_shot(s: &x3d_scene::Scene, gpu: &Gpu, shadows: bool) -> RgbaImage {
    let settings = RenderSettings { shadows, ..Default::default() };
    render_scene(gpu, s, &Shot { size: (64, 64), msaa: 1, settings, viewpoint: Some(1) }).unwrap()
}

#[test]
fn a_spot_light_with_shadows_true_casts_a_perspective_shadow() {
    let Some(gpu) = gpu() else { return };
    // Spot at (1.5, 3, 0) aimed at the floor's centre-left: the slab's
    // shadow lands at x ≈ −1.5 (half a unit wide).
    let s = shadow_stage(
        "<SpotLight location='1.5 3 0' direction='-0.5 -1 0' cutOffAngle='1.2' beamWidth='1.2' shadows='true' radius='20'/>",
    );
    let (plain, shadowed) = (shadow_shot(&s, &gpu, false), shadow_shot(&s, &gpu, true));
    let at = |img: &RgbaImage, x: u32| img.get_pixel(x, 32).0[0];
    // x = −1.5 → pixel 32 − 1.5·12.9 ≈ 13. x = +2.5 (open floor) → 64-ish edge, use 56.
    assert!(at(&plain, 13) > 60, "lit without shadows: {}", at(&plain, 13));
    assert!(at(&shadowed, 13) < at(&plain, 13) / 4, "shadowed: {} vs {}", at(&shadowed, 13), at(&plain, 13));
    assert_eq!(at(&plain, 56), at(&shadowed, 56), "open floor unchanged");
}

#[test]
fn a_point_light_casts_into_its_cube_faces() {
    let Some(gpu) = gpu() else { return };
    // Point light right above the slab: the shadow is straight below (−Y face)
    // and is wider than the slab (perspective).
    let s = shadow_stage("<PointLight location='0 2 0' shadows='true' radius='20'/>");
    let (plain, shadowed) = (shadow_shot(&s, &gpu, false), shadow_shot(&s, &gpu, true));
    // Seen from above the slab hides x∈[−0.5,0.5]; its shadow spans ±1 on the
    // floor (similar triangles: 0.5·2/1). Sample x = 0.75 → pixel ≈ 42.
    let at = |img: &RgbaImage, x: u32| img.get_pixel(x, 32).0[0];
    assert!(at(&plain, 42) > 60, "lit: {}", at(&plain, 42));
    assert!(at(&shadowed, 42) < at(&plain, 42) / 4, "in the cone of shadow: {}", at(&shadowed, 42));
    assert_eq!(at(&plain, 60), at(&shadowed, 60), "beyond the shadow: unchanged");
}

#[test]
fn shadow_intensity_scales_the_darkening() {
    let Some(gpu) = gpu() else { return };
    let light = |k: &str| {
        format!("<DirectionalLight direction='1 -1 0' ambientIntensity='0' shadows='true' shadowIntensity='{k}'/>")
    };
    let full = shadow_shot(&shadow_stage(&light("1")), &gpu, true).get_pixel(45, 32).0[0];
    let half = shadow_shot(&shadow_stage(&light("0.5")), &gpu, true).get_pixel(45, 32).0[0];
    let open = shadow_shot(&shadow_stage(&light("1")), &gpu, true).get_pixel(12, 32).0[0];
    assert!(full < 30, "full shadow: {full}");
    // X3D: lit = mix(1, shadow, intensity) → about half the open floor's light.
    let expected = (f32::from(open) + f32::from(full)) / 2.0;
    assert!((f32::from(half) - expected).abs() < 12.0, "half: {half}, expected ≈ {expected}");
}

#[test]
fn gpu_culling_draws_the_same_picture_and_skips_what_is_behind() {
    let Some(gpu) = gpu() else { return };
    if !gpu.budget.gpu_cull {
        eprintln!("SKIPPED: budget has no GPU culling on {}", gpu.describe());
        return;
    }
    // One box in view, one behind the camera.
    let s = scene(
        "<Shape><Appearance><Material diffuseColor='1 0 0'/></Appearance><Box/></Shape>
         <Transform translation='0 0 20'><Shape><Appearance><Material diffuseColor='0 1 0'/></Appearance><Box/></Shape></Transform>",
    );
    let shot = |cull| Shot {
        size: (64, 64),
        msaa: 1,
        settings: RenderSettings { cull, ..Default::default() },
        viewpoint: None,
    };
    let cpu = render_scene(&gpu, &s, &shot(x3d_render::cull::Cull::Cpu)).unwrap();
    let gpu_img = render_scene(&gpu, &s, &shot(x3d_render::cull::Cull::Gpu)).unwrap();
    assert_eq!(cpu, gpu_img, "same pixels either way");

    let mut r = Renderer::new(&gpu, wgpu::TextureFormat::Rgba8Unorm, 1);
    r.set_scene(&s);
    let target = x3d_render::headless::Offscreen::new(&gpu.device, (64, 64), r.color_format());
    let cam = x3d_render::headless::camera_for(&s, None);
    let st = r.render(&target.view, (64, 64), &cam, &shot(x3d_render::cull::Cull::Cpu).settings);
    assert_eq!((st.draw_calls, st.culled), (1, 1), "CPU: one drawn, one culled");
    let st = r.render(&target.view, (64, 64), &cam, &shot(x3d_render::cull::Cull::Gpu).settings);
    assert_eq!(st.draw_calls, 2, "GPU: both issued (indirect)…");
    assert_eq!(r.gpu_visible(), Some(1), "…and the cull pass kept one");
}

#[test]
fn timestamps_time_each_pass_that_ran() {
    let Some(gpu) = gpu() else { return };
    if !gpu.budget.timestamps {
        eprintln!("SKIPPED: no TIMESTAMP_QUERY on {}", gpu.describe());
        return;
    }
    let s = scene("<Shape><Appearance><Material/></Appearance><Box/></Shape><DirectionalLight/>");
    let mut r = Renderer::new(&gpu, wgpu::TextureFormat::Rgba8Unorm, 1);
    r.set_scene(&s);
    let target = x3d_render::headless::Offscreen::new(&gpu.device, (64, 64), r.color_format());
    let cam = x3d_render::headless::camera_for(&s, None);
    let settings = RenderSettings { shadows: true, ..Default::default() };
    r.render(&target.view, (64, 64), &cam, &settings);
    let t = r.wait_gpu_times().expect("a readback");
    use x3d_render::timing::{BLOOM, SCENE, SHADOW};
    assert!(t[SCENE].is_some_and(|ms| ms > 0.0), "{t:?}");
    assert!(t[SHADOW].is_some_and(|ms| ms > 0.0), "{t:?}");
    assert_eq!(t[BLOOM], None, "bloom didn't run, so it is null, not 0");
}

#[test]
fn a_weak_gpu_profile_degrades_instead_of_failing() {
    let Ok(weak) = Gpu::headless_with(x3d_render::Backend::Auto, x3d_render::Profile::Weak) else { return };
    assert!(!weak.budget.hdr && !weak.budget.gpu_cull && !weak.budget.timestamps, "{}", weak.budget);
    let s = shadow_stage("<DirectionalLight direction='1 -1 0' ambientIntensity='0'/>");
    let everything = RenderSettings {
        shadows: true,
        tone_map: x3d_render::ToneMap::Aces,
        bloom: 0.3,
        cull: x3d_render::cull::Cull::Gpu,
        ..Default::default()
    };
    let ldr = RenderSettings { shadows: true, ..Default::default() };
    let shot = |settings| Shot { size: (64, 64), msaa: 4, settings, viewpoint: Some(1) };
    let a = render_scene(&weak, &s, &shot(everything)).unwrap();
    let b = render_scene(&weak, &s, &shot(ldr)).unwrap();
    assert_eq!(a, b, "HDR and GPU culling fall back; shadows still work with one cascade");
    assert!(a.get_pixel(45, 32).0[0] < 30, "shadow present on the weak profile");
}

#[test]
fn a_lost_device_is_noticed_not_crashed() {
    let Some(gpu) = gpu() else { return };
    assert!(!gpu.is_lost());
    gpu.device.destroy();
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    assert!(gpu.is_lost(), "the device-lost callback set the flag");
}

#[test]
fn hostile_scenes_render_without_panics_or_validation_errors() {
    let Some(gpu) = gpu() else { return };
    // Each must either fail to load with an Err, or render cleanly.
    let cases = [
        (
            "NaN coordinates",
            "<Shape><IndexedFaceSet coordIndex='0 1 2'><Coordinate point='nan 0 0, 1 inf 0, 0 1 -inf'/></IndexedFaceSet></Shape>",
        ),
        (
            "indices out of range",
            "<Shape><IndexedFaceSet coordIndex='0 1 99 -1 5 6 7'><Coordinate point='0 0 0, 1 0 0, 0 1 0'/></IndexedFaceSet></Shape>",
        ),
        (
            "empty geometry",
            "<Shape><IndexedFaceSet/></Shape><Shape><PointSet/></Shape><Shape><IndexedLineSet coordIndex=''/></Shape>",
        ),
        ("huge numbers", "<Transform scale='1e38 1e38 1e38'><Shape><Box size='1e30 1e30 1e30'/></Shape></Transform>"),
        (
            "zero scale",
            "<Transform scale='0 0 0'><Shape><Sphere/></Shape></Transform><DirectionalLight direction='0 0 0' shadows='true'/>",
        ),
        (
            "degenerate lights",
            "<SpotLight cutOffAngle='0' shadows='true'/><PointLight radius='-5' shadows='true'/><Shape><Box/></Shape>",
        ),
        ("20 point lights asking for shadows", &"<PointLight shadows='true'/>".repeat(20)),
        (
            "bad shader",
            "<Shape><Appearance><ComposedShader language='GLSL'><ShaderPart type='VERTEX'>void main( {</ShaderPart><ShaderPart type='FRAGMENT'>nonsense</ShaderPart></ComposedShader></Appearance><Box/></Shape>",
        ),
        (
            "route loop",
            "<TimeSensor DEF='T' loop='true'/><ScalarInterpolator DEF='S' key='0 1' keyValue='0 1'/><ROUTE fromNode='T' fromField='fraction_changed' toNode='S' toField='set_fraction'/><ROUTE fromNode='S' fromField='value_changed' toNode='T' toField='set_startTime'/>",
        ),
    ];
    for (name, body) in cases {
        let Ok(mut s) = x3d_scene::parse_str(&format!("<X3D><Scene>{body}</Scene></X3D>")) else { continue };
        s.advance(0.0);
        s.advance(2.5);
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let settings =
            RenderSettings { shadows: true, tone_map: x3d_render::ToneMap::Aces, bloom: 0.2, ..Default::default() };
        let r = render_scene(&gpu, &s, &Shot { size: (32, 32), msaa: 4, settings, viewpoint: None });
        let err = pollster::block_on(scope.pop());
        assert!(r.is_ok(), "{name}: {r:?}");
        assert!(err.is_none(), "{name}: wgpu validation error {err:?}");
    }
}

#[test]
fn an_uncaptured_validation_error_is_logged_not_fatal() {
    let Some(gpu) = gpu() else { return };
    // A copy past the end of a buffer is a validation error. wgpu's default
    // handler would panic; ours logs (Gpu::new installs it).
    let a = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&a, 0, &b, 0, 64);
    gpu.queue.submit([enc.finish()]);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    // Still alive and usable:
    let s = scene("<Shape><Appearance><UnlitMaterial emissiveColor='0 1 0'/></Appearance><Box size='3 3 3'/></Shape>");
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert!(close(center(&img), [0, 255, 0, 255], 1));
}
