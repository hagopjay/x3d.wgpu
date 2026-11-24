//! Fog and PixelTexture: found missing by the cross-engine render oracle (x3d.loom/x3d.render-oracle).

use x3d_scene::parse_str;

fn x3d(body: &str) -> String {
    format!("<X3D><Scene>{body}</Scene></X3D>")
}

#[test]
fn fog_is_read_and_range_zero_turns_it_off() {
    let s = parse_str(&x3d("<Fog color='0.7 0.75 0.85' fogType='LINEAR' visibilityRange='14'/><Shape><Box/></Shape>"))
        .unwrap();
    let f = s.fog.expect("a Fog with a visibilityRange");
    assert!(!f.exponential && (f.visibility_range - 14.0).abs() < 1e-6 && (f.color.y - 0.75).abs() < 1e-6);
    let e = parse_str(&x3d("<Fog fogType='EXPONENTIAL' visibilityRange='5'/><Shape><Box/></Shape>")).unwrap();
    assert!(e.fog.unwrap().exponential);
    let off = parse_str(&x3d("<Fog visibilityRange='0'/><Shape><Box/></Shape>")).unwrap();
    assert!(off.fog.is_none());
    let none = parse_str(&x3d("<Shape><Box/></Shape>")).unwrap();
    assert!(none.fog.is_none());
}

#[test]
fn a_pixel_texture_becomes_a_nearest_filtered_png() {
    let s = parse_str(&x3d(
        "<Shape><Appearance><Material/><PixelTexture image='2 2 3 0xff0000 0x00ff00 0x0000ff 0xffffff' repeatS='false'/></Appearance><Box/></Shape>",
    ))
    .unwrap();
    let t = s.draws[0].material.textures.base.as_ref().expect("the PixelTexture is the base texture");
    assert!(t.nearest && !t.repeat_s && t.repeat_t);
    let img = image::load_from_memory(t.inline.as_ref().unwrap()).unwrap().to_rgba8();
    assert_eq!(img.dimensions(), (2, 2));
    // SFImage: the first pixel is the picture's lower left, so it is the file's bottom-left pixel.
    assert_eq!(img.get_pixel(0, 1).0, [255, 0, 0, 255]);
    assert_eq!(img.get_pixel(1, 0).0, [255, 255, 255, 255]);
}

#[test]
fn a_pixel_texture_with_one_component_is_grey_and_a_short_image_warns() {
    let s = parse_str(&x3d(
        "<Shape><Appearance><Material/><PixelTexture image='1 2 1 0x00 0xff'/></Appearance><Box/></Shape>",
    ))
    .unwrap();
    let t = s.draws[0].material.textures.base.as_ref().unwrap();
    let img = image::load_from_memory(t.inline.as_ref().unwrap()).unwrap().to_rgba8();
    assert_eq!(img.get_pixel(0, 1).0, [0, 0, 0, 255]);
    assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255, 255]);
    let short = parse_str(&x3d(
        "<Shape><Appearance><Material/><PixelTexture image='2 2 3 0xff0000'/></Appearance><Box/></Shape>",
    ))
    .unwrap();
    assert!(short.draws[0].material.textures.base.is_none());
}
