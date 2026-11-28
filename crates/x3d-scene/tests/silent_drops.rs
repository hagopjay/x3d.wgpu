//! The loss report counts every node it does not draw (found by the 260-node audit).

use x3d_scene::parse_str;

fn scene(body: &str) -> x3d_scene::Scene {
    parse_str(&format!("<X3D><Scene>{body}</Scene></X3D>")).unwrap()
}

#[test]
fn unseen_children_of_an_appearance_are_counted() {
    let s = scene("<Shape><Appearance><FillProperties filled='false'/></Appearance><Box/></Shape>");
    assert_eq!(s.report.unsupported.get("FillProperties"), Some(&1));
}

#[test]
fn sensors_and_chasers_are_counted_as_not_run() {
    let s = scene("<TouchSensor/><PositionChaser/><Shape><Box/></Shape>");
    assert!(s.report.ignored.keys().any(|k| k.starts_with("TouchSensor ")));
    assert!(s.report.ignored.keys().any(|k| k.starts_with("PositionChaser ")));
}

#[test]
fn nodes_the_loader_already_reports_are_not_counted_twice() {
    let s = scene("<Shape><Text string='hi'/></Shape>");
    assert_eq!(s.report.unsupported.get("Text"), Some(&1));
}

#[test]
fn supported_scenes_stay_lossless() {
    let s = scene("<Shape><Appearance><Material/></Appearance><Box/></Shape>");
    assert!(s.report.is_lossless());
}
