//! Golden images: every sample scene (and the lighting extras) rendered and
//! diffed against a PNG blessed on a named adapter.
//!
//! - `X3D_REQUIRE_GPU=1` turns "no adapter" from a skip into a failure (CI sets it).
//! - `X3D_BLESS=1` rewrites the goldens and `tests/golden/SOURCE.txt`.
//! - On failure, `target/golden/<name>.{got,diff}.png` and `report.txt` explain it.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use x3d_render::diff::{self, Tolerance};
use x3d_render::{Gpu, RenderSettings, Shot, ToneMap, render_scene};

const SIZE: (u32, u32) = (192, 108);

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn gpu() -> Option<Gpu> {
    x3d_render::validation::install("warn");
    match Gpu::headless(x3d_render::Backend::Auto) {
        Ok(g) => Some(g),
        Err(e) if std::env::var_os("X3D_REQUIRE_GPU").is_some() => {
            panic!("X3D_REQUIRE_GPU is set but there is no adapter: {e}")
        }
        Err(e) => {
            eprintln!("SKIPPED golden images (no adapter): {e}");
            None
        }
    }
}

/// (golden name, scene file, settings)
fn cases() -> Vec<(String, PathBuf, RenderSettings)> {
    let mut cases: Vec<_> = std::fs::read_dir(root().join("scenes"))
        .expect("scenes/")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "x3d"))
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), p, RenderSettings::default()))
        .collect();
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    let extras = RenderSettings { shadows: true, tone_map: ToneMap::Aces, bloom: 0.15, ..Default::default() };
    cases.push(("05_still_life+extras".into(), root().join("scenes/05_still_life.x3d"), extras));
    cases
}

#[test]
fn sample_scenes_match_their_goldens() {
    let Some(gpu) = gpu() else { return };
    let info = gpu.adapter.get_info();
    let adapter = format!("{} {} {}", info.name, info.driver, info.driver_info);
    let dir = root().join("crates/x3d-render/tests/golden");
    let table = Tolerance::parse_table(&std::fs::read_to_string(dir.join("TOLERANCE")).expect("TOLERANCE"));
    let tol = Tolerance::pick(&table, &adapter).expect("a `*` row").clone();
    let bless = std::env::var_os("X3D_BLESS").is_some();
    let out = root().join("target/golden");
    std::fs::create_dir_all(&out).unwrap();

    let mut report = format!(
        "adapter: {}\ntolerance: {} (max_delta {}, max_bad {}‰) — {}\n\n{:<28} {:>9} {:>6} {:>9} {:>8}  verdict\n",
        x3d_render::gpu::describe(&info),
        tol.adapter,
        tol.max_delta,
        tol.max_bad_permille,
        tol.why,
        "golden",
        "mean|Δ|",
        "max|Δ|",
        "bad px",
        "bad ‰"
    );
    let mut failed = Vec::new();
    for (name, path, settings) in cases() {
        let scene = x3d_scene::load(&path).unwrap();
        let shot = Shot { size: SIZE, msaa: 4, settings, viewpoint: None };
        let got = render_scene(&gpu, &scene, &shot).unwrap();
        let golden_path = dir.join(format!("{name}.png"));
        if bless {
            got.save(&golden_path).unwrap();
            continue;
        }
        let golden =
            image::open(&golden_path).unwrap_or_else(|e| panic!("{name}: {e} (X3D_BLESS=1 makes it)")).to_rgba8();
        let d = diff::diff(&golden, &got, tol.max_delta).expect("same size");
        let ok = tol.accepts(&d);
        writeln!(
            report,
            "{name:<28} {:>9.4} {:>6} {:>9} {:>8.3}  {}",
            d.mean_abs,
            d.max_abs,
            d.bad_pixels,
            d.bad_permille(),
            if ok { "ok" } else { "FAIL" }
        )
        .unwrap();
        if !ok {
            got.save(out.join(format!("{name}.got.png"))).unwrap();
            diff::heatmap(&golden, &got, tol.max_delta).save(out.join(format!("{name}.diff.png"))).unwrap();
            failed.push(name);
        }
    }
    if bless {
        let source = format!(
            "Blessed with X3D_BLESS=1 on: {}\nsize {}x{}, msaa 4, x3d-render {}\n",
            x3d_render::gpu::describe(&info),
            SIZE.0,
            SIZE.1,
            env!("CARGO_PKG_VERSION")
        );
        std::fs::write(dir.join("SOURCE.txt"), source).unwrap();
        eprintln!("blessed goldens on {adapter}");
        return;
    }
    std::fs::write(out.join("report.txt"), &report).unwrap();
    eprintln!("{report}");
    assert!(failed.is_empty(), "golden mismatches: {failed:?} (see target/golden/)");
    x3d_render::validation::assert_clean("golden images");
}

#[test]
fn sample_scenes_are_still_what_x3d_mcp_validated() {
    // validators/MANIFEST.json holds each scene's sha256 at validation time
    // (conformance/validate.sh). An edited scene must be re-validated.
    let manifest = std::fs::read_to_string(root().join("conformance/validators/MANIFEST.json")).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    for (file, v) in manifest["files"].as_object().unwrap() {
        use sha2::Digest;
        let bytes = std::fs::read(root().join(file)).unwrap();
        let sha: String = sha2::Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            v["sha256"].as_str(),
            Some(sha.as_str()),
            "{file} changed since validation: run conformance/validate.sh"
        );
        assert_eq!(v["xsd_valid"], serde_json::json!(true), "{file}");
        assert_eq!(v["semantic_errors"], serde_json::json!(0), "{file}");
    }
}
