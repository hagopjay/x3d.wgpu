//! Conformance rows from `conformance/expected.json`: X_ITE measurements
//! recorded by the Loom, scenes validated by x3d_mcp (see that folder).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use x3d_render::{Gpu, conform};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

// A libtest thread limit in CI alone does not protect `cargo test` locally.
// Keep device creation, rendering AND teardown exclusive within this binary.
// The runner's native crash has not been reduced to a particular driver bug;
// avoiding overlapping devices is a bounded workaround, not that diagnosis.
static GPU_TEST: Mutex<()> = Mutex::new(());

#[derive(Debug)]
struct TestGpu {
    // Struct fields drop in declaration order: drop all GPU handles before
    // releasing the lock and allowing another test to create its device.
    gpu: Gpu,
    _serial: MutexGuard<'static, ()>,
}

impl std::ops::Deref for TestGpu {
    type Target = Gpu;

    fn deref(&self) -> &Gpu {
        &self.gpu
    }
}

fn gpu(test: &str) -> Option<TestGpu> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("x3d_render=info"))
        .is_test(true)
        .try_init();
    let serial = GPU_TEST.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    eprintln!("conformance test={test}: creating device (exclusive fixture)");
    match Gpu::headless(x3d_render::Backend::Auto) {
        Ok(gpu) => {
            eprintln!("conformance test={test}: adapter={}", gpu.describe());
            // The interactive renderer logs validation failures and continues;
            // a conformance test must fail rather than accept missing work.
            gpu.device.on_uncaptured_error(Arc::new(|error| panic!("conformance wgpu validation error: {error}")));
            Some(TestGpu { gpu, _serial: serial })
        }
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
    let Some(gpu) = gpu("every_measured_row_passes_and_names_its_source") else { return };
    let r = conform::run(&gpu, &root(), &root().join("conformance")).unwrap();
    eprintln!("{}", r.table());
    assert!(r.passed(), "{}", r.table());
    assert!(r.rows.iter().filter(|r| r.verdict == "pass").count() >= 30, "{}", r.table());
    assert_eq!(r.json["summary"]["not_run"], 0, "every X_ITE row is measured now (EnvironmentLight landed in 0.5)");
    for row in r.json["rows"].as_array().unwrap() {
        assert!(row["expected_from"]["source"]["record"].is_string(), "row without a record: {row}");
    }
}

#[test]
fn a_scene_edited_after_validation_is_refused() {
    let Some(gpu) = gpu("a_scene_edited_after_validation_is_refused") else { return };
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

#[test]
fn the_residual_against_x_ite_is_the_encode_curve() {
    // RESEARCH-ibl.md: with X_ITE's own 2.2 encode, our shading lands within
    // one byte of every X_ITE row; the sRGB default sits up to 2 above.
    let Some(gpu) = gpu("the_residual_against_x_ite_is_the_encode_curve") else { return };
    let r = conform::run(&gpu, &root(), &root().join("conformance")).unwrap();
    for row in &r.rows {
        let (Some(g), e) = (row.got_gamma22, row.expected) else { continue };
        let d = (0..3).map(|c| g[c].abs_diff(e[c])).max().unwrap();
        assert!(d <= 1, "{}: γ2.2 {:?} vs X_ITE {:?}", row.id, g, e);
    }
}
