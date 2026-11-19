//! Conformance rows from `conformance/expected.json`: X_ITE measurements
//! recorded by the Loom, scenes validated by x3d_mcp (see that folder).

use std::path::{Path, PathBuf};

use x3d_render::{Gpu, conform};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn gpu() -> Option<Gpu> {
    match Gpu::headless(x3d_render::Backend::Auto) {
        Ok(g) => Some(g),
        Err(e) if std::env::var_os("X3D_REQUIRE_GPU").is_some() => {
            panic!("X3D_REQUIRE_GPU is set but there is no adapter: {e}")
        }
        Err(e) => {
            eprintln!("SKIPPED conformance (no adapter): {e}");
            None
        }
    }
}

#[test]
fn every_measured_row_passes_and_names_its_source() {
    let Some(gpu) = gpu() else { return };
    let r = conform::run(&gpu, &root(), &root().join("conformance")).unwrap();
    eprintln!("{}", r.table());
    assert!(r.passed(), "{}", r.table());
    assert!(r.rows.iter().filter(|r| r.verdict == "pass").count() >= 6);
    for row in r.json["rows"].as_array().unwrap() {
        assert!(row["expected_from"]["source"]["record"].is_string(), "row without a record: {row}");
    }
}

#[test]
fn a_scene_edited_after_validation_is_refused() {
    let Some(gpu) = gpu() else { return };
    // Copy conformance/ and change one scene by a byte: its sha256 no longer
    // matches what x3d_mcp validated, so the row must not pass.
    let tmp = std::env::temp_dir().join(format!("x3d-conform-{}", std::process::id()));
    let dir = tmp.join("conformance");
    std::fs::create_dir_all(dir.join("scenes")).unwrap();
    std::fs::create_dir_all(dir.join("validators")).unwrap();
    for f in ["expected.json", "validators/MANIFEST.json"] {
        std::fs::copy(root().join("conformance").join(f), dir.join(f)).unwrap();
    }
    for e in std::fs::read_dir(root().join("conformance/scenes")).unwrap() {
        let p = e.unwrap().path();
        std::fs::copy(&p, dir.join("scenes").join(p.file_name().unwrap())).unwrap();
    }
    let victim = dir.join("scenes/exp2_a_unlit_default.x3d");
    let text = std::fs::read_to_string(&victim).unwrap().replace("0.5 0.5 0.5", "0.5 0.5 0.51");
    std::fs::write(&victim, text).unwrap();
    let r = conform::run(&gpu, &tmp, &dir).unwrap();
    let row = r.rows.iter().find(|r| r.id == "exp2_a_unlit_default").unwrap();
    assert_eq!(row.verdict, "stale validation", "{}", r.table());
    assert!(!r.passed());
    std::fs::remove_dir_all(&tmp).ok();
}
