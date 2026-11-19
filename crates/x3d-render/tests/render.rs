//! GPU integration tests: render tiny scenes and check actual pixels.
//!
//! They need *some* GPU adapter. On a machine without one (CI, containers),
//! install a software Vulkan driver (`apt install mesa-vulkan-drivers`); if
//! no adapter exists at all the GPU tests print a note and pass.

use image::{Rgba, RgbaImage};
use x3d_render::renderer::BUILTIN_SHADER;
use x3d_render::uniforms::{DrawUniform, FrameUniform};
use x3d_render::{ColorMode, Gpu, RenderSettings, Renderer, Shot, render_scene};

fn gpu() -> Option<Gpu> {
    match Gpu::headless(x3d_render::Backend::Auto) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn scene(body: &str) -> x3d_scene::Scene {
    x3d_scene::parse_str(&format!(
        "<X3D><Scene><Viewpoint position='0 0 5'/>{body}</Scene></X3D>"
    ))
    .unwrap()
}

fn shot(settings: RenderSettings) -> Shot {
    Shot {
        size: (64, 64),
        msaa: 1,
        settings,
        viewpoint: None,
    }
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
    assert_eq!(
        img.get_pixel(0, 0),
        &Rgba([0, 0, 255, 255]),
        "corner shows the Background"
    );
}

#[test]
fn unlit_colours_reach_the_screen_exactly() {
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Shape><Appearance><UnlitMaterial emissiveColor='0.5 0.25 1'/></Appearance><Box/></Shape>",
    );
    let img = render_scene(&gpu, &s, &shot(RenderSettings::default())).unwrap();
    assert!(
        close(center(&img), [128, 64, 255, 255], 1),
        "{:?}",
        center(&img)
    );
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
    let linear_settings = RenderSettings {
        color_mode: ColorMode::Linear,
        ..Default::default()
    };
    let linear = center(&render_scene(&gpu, &s, &shot(linear_settings)).unwrap());
    // display: 0.8 × 0.707 = 0.566 → 144. linear: srgb(lin(0.8) × 0.707) ≈ 0.69 → 175.
    assert!(
        close(display, [144, 144, 144, 255], 2),
        "display {display:?}"
    );
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
    assert!(
        close(c, [255, 128, 128, 255], 2),
        "50 % white over red should be pink: {c:?}"
    );
}

#[test]
fn a_broken_shader_is_rejected_and_the_old_one_keeps_working() {
    let Some(gpu) = gpu() else { return };
    let s = scene(
        "<Shape><Appearance><UnlitMaterial emissiveColor='0 1 0'/></Appearance><Box/></Shape>",
    );
    let mut renderer = Renderer::new(&gpu, wgpu::TextureFormat::Rgba8Unorm, 1);
    renderer.set_scene(&s);

    let err = renderer.reload_shader("fn this is not wgsl").unwrap_err();
    assert!(!err.is_empty());

    // A shader that parses but whose bindings don't match the Rust side.
    let mismatched = BUILTIN_SHADER.replace("@group(1) @binding(0)", "@group(1) @binding(7)");
    assert!(renderer.reload_shader(&mismatched).is_err());

    let img = render_with(&gpu, &mut renderer, &s);
    assert!(
        close(center(&img), [0, 255, 0, 255], 1),
        "old shader still draws"
    );

    // And a good edit takes effect.
    let magenta = BUILTIN_SHADER.replace(
        "return vec4f(rgb, alpha);",
        "return vec4f(1.0, 0.0, 1.0, 1.0);",
    );
    renderer.reload_shader(&magenta).unwrap();
    let img = render_with(&gpu, &mut renderer, &s);
    assert!(close(center(&img), [255, 0, 255, 255], 1));
}

fn render_with(gpu: &Gpu, renderer: &mut Renderer, s: &x3d_scene::Scene) -> RgbaImage {
    let target =
        x3d_render::headless::Offscreen::new(&gpu.device, (32, 32), renderer.color_format());
    let camera = x3d_render::headless::camera_for(s, None);
    renderer.render(&target.view, (32, 32), &camera, &RenderSettings::default());
    target.read(gpu).unwrap()
}

#[test]
fn odd_widths_read_back_correctly() {
    // 37 px × 4 bytes = 148 bytes per row, padded to 256 on the GPU side.
    let Some(gpu) = gpu() else { return };
    let s = scene("<Background skyColor='1 0.5 0'/>");
    let img = render_scene(
        &gpu,
        &s,
        &Shot {
            size: (37, 5),
            ..shot(RenderSettings::default())
        },
    )
    .unwrap();
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
                naga::TypeInner::Struct { span, .. } if ty.name.as_deref() == Some(name) => {
                    Some(*span as usize)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("struct {name} not in the shader"))
    };
    assert_eq!(size_of_wgsl("Frame"), size_of::<FrameUniform>());
    assert_eq!(size_of_wgsl("Draw"), size_of::<DrawUniform>());
    assert_eq!(
        size_of_wgsl("Light"),
        size_of::<x3d_render::uniforms::GpuLight>()
    );

    // Full validation, as wgpu would do at runtime.
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
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
    assert!(
        close(c, [78, 78, 78, 255], 2),
        "PBR centre {c:?}, X_ITE gives 78"
    );
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
    assert!(
        close(c, [255, 0, 0, 255], 8),
        "expected the top-left (red) texel, got {c:?}"
    );
}
