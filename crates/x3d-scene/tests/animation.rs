//! The event system: TimeSensors → interpolators → targets, through ROUTEs.

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use x3d_scene::parse_str;

fn x3d(body: &str) -> String {
    format!("<X3D><Scene>{body}</Scene></X3D>")
}

/// A Transform that turns once around Y every 4 s.
const SPINNER: &str = "
  <Transform DEF='Spin'><Transform translation='2 0 0'><Shape><Box/></Shape></Transform></Transform>
  <TimeSensor DEF='Clock' cycleInterval='4' loop='true'/>
  <OrientationInterpolator DEF='Turn' key='0 0.25 0.5 0.75 1'
     keyValue='0 1 0 0, 0 1 0 1.5708, 0 1 0 3.1416, 0 1 0 4.7124, 0 1 0 6.2832'/>
  <ROUTE fromNode='Clock' fromField='fraction_changed' toNode='Turn' toField='set_fraction'/>
  <ROUTE fromNode='Turn' fromField='value_changed' toNode='Spin' toField='set_rotation'/>";

#[test]
fn a_spinning_transform_moves_its_children() {
    let mut s = parse_str(&x3d(SPINNER)).unwrap();
    assert_eq!(s.report.stats.routes, 2);
    let at = |s: &x3d_scene::Scene| s.draws[0].transform.transform_point3(Vec3::ZERO);
    assert!((at(&s) - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4);

    let changes = s.advance(1.0); // a quarter turn: +X goes to −Z
    assert!(changes.transforms && changes.events == 2, "{changes:?}");
    assert!((at(&s) - Vec3::new(0.0, 0.0, -2.0)).length() < 1e-3, "{}", at(&s));

    s.advance(4.0); // a full cycle later: back where it started
    assert!((at(&s) - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-3, "{}", at(&s));
}

#[test]
fn one_material_event_recolours_every_shape_that_uses_it() {
    let mut s = parse_str(&x3d(
        "<Shape><Appearance><UnlitMaterial DEF='Beam' emissiveColor='1 0 0'/></Appearance><Box/></Shape>
         <Shape><Appearance><UnlitMaterial USE='Beam'/></Appearance><Sphere/></Shape>
         <TimeSensor DEF='T' cycleInterval='2' loop='true'/>
         <ColorInterpolator DEF='C' key='0 1' keyValue='1 0 0, 0 0 1'/>
         <ROUTE fromNode='T' fromField='fraction_changed' toNode='C' toField='set_fraction'/>
         <ROUTE fromNode='C' fromField='value_changed' toNode='Beam' toField='set_emissiveColor'/>",
    ))
    .unwrap();
    s.advance(1.0); // halfway: HSV blend of red → blue is magenta
    for d in &s.draws {
        assert!((d.material.emissive - Vec3::new(1.0, 0.0, 1.0)).length() < 1e-4, "{}", d.material.emissive);
    }
}

#[test]
fn a_sequencer_flips_a_switch() {
    let mut s =
        parse_str(&x3d("<Switch DEF='Which' whichChoice='0'><Shape><Box/></Shape><Shape><Sphere/></Shape></Switch>
         <TimeSensor DEF='T' cycleInterval='2' loop='true'/>
         <IntegerSequencer DEF='Seq' key='0 0.5' keyValue='0 1'/>
         <ROUTE fromNode='T' fromField='fraction_changed' toNode='Seq' toField='set_fraction'/>
         <ROUTE fromNode='Seq' fromField='value_changed' toNode='Which' toField='set_whichChoice'/>"))
        .unwrap();
    assert_eq!(s.draws.len(), 2, "an animated Switch keeps every child");
    let shown = |s: &x3d_scene::Scene| -> Vec<bool> { s.draws.iter().map(|d| s.switched_on(&d.switches)).collect() };
    assert_eq!(shown(&s), [true, false]);
    s.advance(1.5); // fraction 0.75 → step 1
    assert_eq!(shown(&s), [false, true]);
}

#[test]
fn routes_fire_once_per_timestamp_so_loops_end() {
    // A → B → A: without the X3D rule this would never stop.
    let mut s = parse_str(&x3d("<TimeSensor DEF='T' loop='true'/>
         <BooleanFilter DEF='A'/><BooleanFilter DEF='B'/>
         <ROUTE fromNode='T' fromField='isActive' toNode='A' toField='set_boolean'/>
         <ROUTE fromNode='A' fromField='inputTrue' toNode='B' toField='set_boolean'/>
         <ROUTE fromNode='B' fromField='inputTrue' toNode='A' toField='set_boolean'/>"))
    .unwrap();
    let changes = s.advance(0.1);
    assert_eq!(changes.events, 3);
}

#[test]
fn one_shot_sensors_that_started_long_ago_stay_quiet() {
    // startTime 0 is 1970: in a browser this cycle ended long ago.
    let mut s = parse_str(&x3d(&SPINNER.replace("loop='true'", "loop='false'"))).unwrap();
    assert_eq!(s.advance(1.0).events, 0);
}

#[test]
fn point_lights_follow_their_animated_location() {
    let mut s =
        parse_str(&x3d("<Transform translation='0 10 0'><PointLight DEF='Torch' location='0 0 0'/></Transform>
         <TimeSensor DEF='T' cycleInterval='1' loop='true'/>
         <PositionInterpolator DEF='P' key='0 1' keyValue='0 0 0, 4 0 0'/>
         <ROUTE fromNode='T' fromField='fraction_changed' toNode='P' toField='set_fraction'/>
         <ROUTE fromNode='P' fromField='value_changed' toNode='Torch' toField='location'/>"))
        .unwrap();
    s.advance(0.5);
    let x3d_scene::LightKind::Point { location, .. } = s.lights[0].kind else { panic!() };
    assert!((location - Vec3::new(2.0, 10.0, 0.0)).length() < 1e-4, "{location}");
}

#[test]
fn inlined_animations_run_per_instance() {
    let dir = std::env::temp_dir().join(format!("x3d-anim-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("spinner.x3d"), x3d(SPINNER)).unwrap();
    std::fs::write(
        dir.join("main.x3d"),
        x3d("<Transform translation='-10 0 0'><Inline url='\"spinner.x3d\"'/></Transform>
             <Transform translation='10 0 0'><Inline url='\"spinner.x3d\"'/></Transform>"),
    )
    .unwrap();
    let mut s = x3d_scene::load(dir.join("main.x3d")).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(s.report.stats.routes, 4, "each Inline brings its own routes");
    s.advance(1.0);
    let at: Vec<Vec3> = s.draws.iter().map(|d| d.transform.transform_point3(Vec3::ZERO)).collect();
    assert!((at[0] - Vec3::new(-10.0, 0.0, -2.0)).length() < 1e-3, "{at:?}");
    assert!((at[1] - Vec3::new(10.0, 0.0, -2.0)).length() < 1e-3, "{at:?}");
}

#[test]
fn advance_is_deterministic() {
    let mut a = parse_str(&x3d(SPINNER)).unwrap();
    let mut b = parse_str(&x3d(SPINNER)).unwrap();
    a.advance(0.3);
    a.advance(2.7);
    b.advance(2.7);
    assert_eq!(a.draws[0].transform, b.draws[0].transform);
    let q = Quat::from_rotation_y(FRAC_PI_2 * 2.7);
    assert!(b.draws[0].transform.to_scale_rotation_translation().1.dot(q).abs() > 0.999);
}

#[test]
fn binding_a_viewpoint_starts_what_its_bind_time_drives() {
    // The Gobbler's pattern: Viewpoint.bindTime → TimeSensor.startTime.
    let mut s = parse_str(&x3d("<Viewpoint DEF='First' position='0 0 5'/>
         <Viewpoint DEF='Second' position='0 0 9'/>
         <Transform DEF='Mover'><Shape><Box/></Shape></Transform>
         <TimeSensor DEF='Intro' cycleInterval='2'/>
         <PositionInterpolator DEF='Slide' key='0 1' keyValue='0 0 0, 4 0 0'/>
         <ROUTE fromNode='Second' fromField='bindTime' toNode='Intro' toField='startTime'/>
         <ROUTE fromNode='Intro' fromField='fraction_changed' toNode='Slide' toField='set_fraction'/>
         <ROUTE fromNode='Slide' fromField='value_changed' toNode='Mover' toField='set_translation'/>"))
    .unwrap();
    let x = |s: &x3d_scene::Scene| s.draws[0].transform.transform_point3(Vec3::ZERO).x;
    s.advance(0.0);
    s.advance(10.0);
    assert!(x(&s).abs() < 1e-4, "nothing moves until Second is bound: {}", x(&s));

    s.bind_viewpoint(1);
    s.advance(10.0); // the bind fires at the next tick
    s.advance(11.0);
    assert!((x(&s) - 2.0).abs() < 1e-3, "half-way through the intro: {}", x(&s));

    s.advance(20.0);
    assert!((x(&s) - 4.0).abs() < 1e-3, "a one-shot sensor ends on fraction 1: {}", x(&s));
}
