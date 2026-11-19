//! Debug helper: which shapes cover pixel (x, y) of a viewpoint's view?
//! cargo run --release -p x3d-render --example whats_at -- scene.x3d VIEWPOINT W H X Y [TIME]
use x3d_render::headless::camera_for;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let opts = x3d_scene::LoadOptions { load_all_inlines: true, ..Default::default() };
    let mut scene = x3d_scene::load_with(&a[1], &opts)?;
    let n = |i: usize| a[i].parse::<f32>().unwrap();
    scene.advance(a.get(7).map_or(0.0, |t| t.parse().unwrap()));
    let cam = camera_for(&scene, Some(a[2].parse()?));
    let (w, h) = (n(3), n(4));
    let vp = cam.projection(w / h) * cam.view();
    let (px, py) = (n(5) / w * 2.0 - 1.0, 1.0 - n(6) / h * 2.0);
    for (i, d) in scene.draws.iter().enumerate() {
        let bad = !d.transform.is_finite() || !d.transform.inverse().transpose().is_finite();
        if bad {
            println!("NON-FINITE {i} {} det {}", d.mesh.label, d.transform.determinant());
        }
    }
    for (i, l) in scene.lights.iter().enumerate() {
        println!("light {i}: on {} intensity {} {:?}", l.on, l.intensity, l.kind);
    }
    for (i, d) in scene.draws.iter().enumerate() {
        let b = d.world_bounds();
        let corners: Vec<glam::Vec3> = (0..8)
            .map(|k| {
                glam::Vec3::new(
                    if k & 1 == 0 { b.min.x } else { b.max.x },
                    if k & 2 == 0 { b.min.y } else { b.max.y },
                    if k & 4 == 0 { b.min.z } else { b.max.z },
                )
            })
            .map(|c| vp.project_point3(c))
            .collect();
        let (lo, hi) =
            corners.iter().fold((glam::Vec3::INFINITY, glam::Vec3::NEG_INFINITY), |(l, h), c| (l.min(*c), h.max(*c)));
        if px >= lo.x && px <= hi.x && py >= lo.y && py <= hi.y && hi.z > 0.0 {
            let m = &d.material;
            println!(
                "{i:>4} {:<38} unlit={} transp={:.2} size {:.0} centre {:.0} switches {:?}",
                d.mesh.label,
                m.unlit,
                m.transparency,
                b.size(),
                b.center(),
                d.switches
            );
        }
    }
    Ok(())
}
