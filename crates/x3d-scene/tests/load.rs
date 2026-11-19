//! Integration tests: they use the crate exactly as a user would
//! (only the public API), from the outside.

use glam::{Mat4, Vec3};
use proptest::prelude::*;
use x3d_scene::{LightKind, Topology, parse_str};

/// Wrap scene content in the X3D boilerplate.
fn x3d(body: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE X3D PUBLIC "ISO//Web3D//DTD X3D 4.0//EN" "https://www.web3d.org/specifications/x3d-4.0.dtd">
<X3D profile='Interchange' version='4.0'><head><meta name='title' content='test'/></head>
<Scene>{body}</Scene></X3D>"#
    )
}

#[test]
fn def_use_builds_geometry_once() {
    let s = parse_str(&x3d(
        "<Shape><Sphere DEF='Ball' radius='0.5'/></Shape>
         <Transform translation='2 0 0'><Shape><Sphere USE='Ball'/></Shape></Transform>",
    ))
    .unwrap();
    assert_eq!(s.draws.len(), 2);
    assert_eq!(s.report.stats.unique_meshes, 1, "USE should share the mesh");
    assert!(std::sync::Arc::ptr_eq(&s.draws[0].mesh, &s.draws[1].mesh));
    assert_eq!(s.draws[1].transform, Mat4::from_translation(Vec3::X * 2.0));
}

#[test]
fn use_of_a_whole_transform() {
    let s = parse_str(&x3d(
        "<Transform DEF='T' translation='0 1 0'><Shape><Box/></Shape></Transform>
         <Transform translation='5 0 0'><Transform USE='T'/></Transform>",
    ))
    .unwrap();
    let positions: Vec<Vec3> = s
        .draws
        .iter()
        .map(|d| d.transform.w_axis.truncate())
        .collect();
    assert_eq!(positions, [Vec3::Y, Vec3::new(5.0, 1.0, 0.0)]);
}

#[test]
fn switch_draws_only_its_choice() {
    let s = parse_str(&x3d(
        "<Switch whichChoice='1'><Shape><Box/></Shape><Shape><Sphere/></Shape><Shape><Cone/></Shape></Switch>
         <Switch><Shape><Box/></Shape></Switch>",
    ))
    .unwrap();
    assert_eq!(s.draws.len(), 1);
    assert!(s.draws[0].mesh.label.starts_with("Sphere"));
}

#[test]
fn nested_transforms_compose_parent_first() {
    let s = parse_str(&x3d(
        "<Transform translation='1 0 0'><Transform scale='2 2 2'><Transform translation='1 0 0'>
           <Shape><Box/></Shape></Transform></Transform></Transform>",
    ))
    .unwrap();
    // 1 + 2·1 = 3 along x.
    let origin = s.draws[0].transform.transform_point3(Vec3::ZERO);
    assert!((origin - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-6);
}

#[test]
fn indexed_face_set_with_per_face_colours() {
    let s = parse_str(&x3d("<Shape><Appearance><Material/></Appearance>
           <IndexedFaceSet coordIndex='0 1 2 3 -1 0 3 4' colorPerVertex='false'>
             <Coordinate point='0 0 0 1 0 0 1 1 0 0 1 0 -1 0.5 0'/>
             <Color color='1 0 0 0 0 1'/>
           </IndexedFaceSet></Shape>"))
    .unwrap();
    let m = &s.draws[0].mesh;
    assert!(m.has_colors);
    assert_eq!(m.primitive_count(), 3); // quad (2) + triangle (1)
    let colours: std::collections::HashSet<[u32; 4]> = m
        .vertices
        .iter()
        .map(|v| v.color.map(f32::to_bits))
        .collect();
    assert_eq!(colours.len(), 2);
}

#[test]
fn lines_and_points_are_unlit() {
    let s = parse_str(&x3d(
        "<Shape><Appearance><Material emissiveColor='1 0 0'/></Appearance>
           <IndexedLineSet coordIndex='0 1 2 -1'><Coordinate point='0 0 0 1 0 0 1 1 0'/></IndexedLineSet></Shape>
         <Shape><PointSet><Coordinate point='0 0 0 1 1 1'/></PointSet></Shape>",
    ))
    .unwrap();
    assert_eq!(s.draws[0].mesh.topology, Topology::Lines);
    assert_eq!(s.draws[0].mesh.primitive_count(), 2);
    assert!(s.draws[0].material.unlit);
    assert_eq!(s.draws[1].mesh.topology, Topology::Points);
    assert_eq!(s.report.stats.points, 2);
}

#[test]
fn lights_are_in_world_space() {
    let s = parse_str(&x3d(
        "<Transform rotation='0 1 0 1.5707963'><DirectionalLight direction='0 0 -1'/></Transform>
         <Transform translation='0 3 0'><PointLight location='1 0 0'/></Transform>
         <PointLight on='false'/>",
    ))
    .unwrap();
    assert_eq!(s.lights.len(), 2, "lights that are off are skipped");
    let LightKind::Directional { direction } = s.lights[0].kind else {
        panic!("not directional")
    };
    assert!((direction - Vec3::NEG_X).length() < 1e-5, "{direction}");
    let LightKind::Point { location, .. } = s.lights[1].kind else {
        panic!("not a point light")
    };
    assert_eq!(location, Vec3::new(1.0, 3.0, 0.0));
}

#[test]
fn navigation_info_and_viewpoint() {
    let s = parse_str(&x3d("<NavigationInfo headlight='false'/>
         <Viewpoint description='Front' position='0 1 5' fieldOfView='0.5'/>"))
    .unwrap();
    assert!(!s.headlight);
    assert_eq!(s.viewpoints[0].description, "Front");
    assert_eq!(s.viewpoints[0].position, Vec3::new(0.0, 1.0, 5.0));
}

#[test]
fn unsupported_nodes_are_reported_not_dropped_silently() {
    let s = parse_str(&x3d(
        "<Shape><Text string='\"hi\"'/></Shape>
         <TimeSensor loop='true'/>
         <Shape><Appearance><ImageTexture url='\"a.png\"'/><Material/></Appearance><Box/></Shape>",
    ))
    .unwrap();
    assert_eq!(s.report.unsupported.get("Text"), Some(&1));
    assert_eq!(s.report.ignored.get("TimeSensor"), Some(&1));
    assert!(
        s.report
            .unsupported
            .keys()
            .any(|k| k.starts_with("ImageTexture"))
    );
    assert_eq!(s.draws.len(), 1, "the box still draws, untextured");
}

#[test]
fn bad_shapes_are_skipped_unless_strict() {
    let doc = x3d("<Shape><Box size='1 two 3'/></Shape><Shape><Sphere/></Shape>");
    let lenient = parse_str(&doc).unwrap();
    assert_eq!(lenient.draws.len(), 1);
    assert_eq!(lenient.report.warnings.len(), 1);
    assert!(
        lenient.report.warnings[0].contains("`two` is not a number"),
        "{:?}",
        lenient.report.warnings
    );

    let strict = x3d_scene::parse_str_with(
        &doc,
        &x3d_scene::LoadOptions {
            strict: true,
            ..Default::default()
        },
    );
    assert!(
        matches!(strict, Err(x3d_scene::Error::Field { line: 4, .. })),
        "{strict:?}"
    );
}

#[test]
fn missing_def_is_an_error_with_the_name() {
    let err = parse_str(&x3d("<Transform USE='Nope'/>")).unwrap_err();
    assert_eq!(err.to_string(), "USE='Nope' has no matching DEF");
}

#[test]
fn classic_encoding_gets_a_helpful_message() {
    let err = parse_str("#X3D V3.3 utf8\nShape { geometry Box {} }").unwrap_err();
    assert!(err.to_string().contains("ClassicVRML"));
}

#[test]
fn elevation_grid_faces_up() {
    let s = parse_str(&x3d(
        "<Shape><ElevationGrid xDimension='3' zDimension='3' height='0 0 0 0 1 0 0 0 0' creaseAngle='3'/></Shape>",
    ))
    .unwrap();
    let m = &s.draws[0].mesh;
    assert_eq!(m.primitive_count(), 8);
    assert!(
        m.vertices.iter().all(|v| v.normal[1] > 0.0),
        "normals point up"
    );
}

#[test]
fn strips_and_fans() {
    let s = parse_str(&x3d(
        "<Shape><TriangleStripSet stripCount='5'><Coordinate point='0 0 0 0 1 0 1 0 0 1 1 0 2 0 0'/></TriangleStripSet></Shape>
         <Shape><IndexedTriangleFanSet index='0 1 2 3 -1'><Coordinate point='0 0 0 1 0 0 1 1 0 0 1 0'/></IndexedTriangleFanSet></Shape>",
    ))
    .unwrap();
    assert_eq!(s.draws[0].mesh.primitive_count(), 3);
    assert_eq!(s.draws[1].mesh.primitive_count(), 2);
    // Strips alternate winding; after the fix-up every triangle faces +Z.
    for d in &s.draws {
        assert!(d.mesh.vertices.iter().all(|v| v.normal[2].abs() > 0.99));
        let n: Vec<f32> = d.mesh.vertices.iter().map(|v| v.normal[2]).collect();
        assert!(
            n.windows(2).all(|w| w[0] == w[1]),
            "consistent facing: {n:?}"
        );
    }
}

#[cfg(feature = "gzip")]
#[test]
fn gzipped_files_load() {
    use std::io::Write;
    let dir = std::env::temp_dir().join(format!("x3d-scene-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("box.x3dz");
    let mut gz = flate2_writer(std::fs::File::create(&path).unwrap());
    gz.write_all(x3d("<Shape><Box/></Shape>").as_bytes())
        .unwrap();
    drop(gz);
    assert_eq!(x3d_scene::load(&path).unwrap().draws.len(), 1);
    std::fs::remove_dir_all(dir).ok();
}

#[cfg(feature = "gzip")]
fn flate2_writer(f: std::fs::File) -> impl std::io::Write {
    // The test borrows flate2 through x3d-scene's own dependency list.
    flate2::write::GzEncoder::new(f, flate2::Compression::fast())
}

// ---------------------------------------------------------------------------
// Property tests: proptest generates hundreds of random inputs and, when
// one fails, *shrinks* it to the smallest failing case.
// ---------------------------------------------------------------------------

proptest! {
    /// Any coordIndex, valid or not, must load without panicking; every
    /// index in the result must point at a real vertex.
    #[test]
    fn indexed_face_sets_never_panic(index in prop::collection::vec(-1i32..12, 0..60)) {
        let coord_index: Vec<String> = index.iter().map(i32::to_string).collect();
        let doc = x3d(&format!(
            "<Shape><IndexedFaceSet coordIndex='{}' creaseAngle='1'>
               <Coordinate point='0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1'/>
             </IndexedFaceSet></Shape>",
            coord_index.join(" ")
        ));
        let scene = parse_str(&doc).unwrap(); // lenient: bad indices skip the shape
        for d in &scene.draws {
            let n = d.mesh.vertices.len() as u32;
            prop_assert!(d.mesh.indices.iter().all(|&i| i < n));
            prop_assert_eq!(d.mesh.indices.len() % 3, 0);
        }
    }

    /// Field parsing is total: it returns Ok or Err, never panics.
    #[test]
    fn field_parsing_never_panics(s in ".{0,40}") {
        let _ = x3d_scene::fields::grouped::<3>(&s);
        let _ = x3d_scene::fields::ints(&s);
    }

    /// Fan triangulation: an n-gon gives n − 2 triangles.
    #[test]
    fn ngon_triangle_count(n in 3usize..40) {
        let pts: Vec<String> = (0..n)
            .map(|i| { let a = i as f32 / n as f32 * std::f32::consts::TAU; format!("{} {} 0", a.cos(), a.sin()) })
            .collect();
        let idx: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let doc = x3d(&format!(
            "<Shape><IndexedFaceSet coordIndex='{}'><Coordinate point='{}'/></IndexedFaceSet></Shape>",
            idx.join(" "), pts.join(", ")
        ));
        prop_assert_eq!(parse_str(&doc).unwrap().report.stats.triangles, n - 2);
    }
}
