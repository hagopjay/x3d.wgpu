//! Conformance receipts: render scenes whose expected pixels came from a
//! named measurement, check each scene was validated, and write a receipt
//! in which every number names its source.
//!
//! Inputs (in `conformance/`):
//! - `expected.json`: rows of (scene, probe, expected RGB, tolerance and the
//!   reason for it, which recorded measurement it came from);
//! - `validators/MANIFEST.json`: each scene's sha256 when x3d_mcp (run by the
//!   Loom's grammar + meaning witnesses) last validated it, and the verdict.
//!
//! A row passes only if the scene is unchanged since validation, valid
//! (XSD and semantics), and every channel is within `max_delta`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::gpu::{Gpu, describe};
use crate::headless::{Shot, render_scene};
use crate::renderer::{Encode, RenderSettings};

#[derive(Debug, Clone)]
pub struct RowResult {
    pub id: String,
    pub probe: String,
    pub expected: [u8; 3],
    pub got: Option<[u8; 3]>,
    pub max_delta: u8,
    pub delta: Option<u8>,
    /// The same row rendered with X_ITE's encode (`--encode gamma22`).
    pub got_gamma22: Option<[u8; 3]>,
    /// "pass", "fail", "invalid scene", "stale validation", "not run"
    pub verdict: String,
    pub source: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct Receipt {
    pub json: Value,
    pub rows: Vec<RowResult>,
}

impl Receipt {
    pub fn passed(&self) -> bool {
        self.rows.iter().all(|r| r.verdict == "pass" || r.verdict == "not run")
    }

    pub fn table(&self) -> String {
        let mut s = format!(
            "{:<30} {:<9} {:>11} {:>11} {:>4} {:>4} {:>11} {:>5}  {:<16} expected from\n",
            "row", "probe", "expected", "got", "|Δ|", "tol", "got γ2.2", "|Δ|", "verdict"
        );
        for r in &self.rows {
            let rgb = |c: [u8; 3]| format!("{},{},{}", c[0], c[1], c[2]);
            writeln!(
                s,
                "{:<30} {:<9} {:>11} {:>11} {:>4} {:>4} {:>11} {:>5}  {:<16} {}",
                r.id,
                r.probe,
                rgb(r.expected),
                r.got.map_or("—".into(), rgb),
                r.delta.map_or("—".into(), |d| d.to_string()),
                r.max_delta,
                r.got_gamma22.map_or("—".into(), rgb),
                r.got_gamma22.map_or("—".into(), |g| (0..3)
                    .map(|c| g[c].abs_diff(r.expected[c]))
                    .max()
                    .unwrap_or(0)
                    .to_string()),
                r.verdict,
                r.source
            )
            .ok();
        }
        s
    }
}

fn sha256(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect())
}

/// Mean RGB of the 5×5 block at the image centre (the X_ITE probe), or the
/// pixel at (2, 2) for `corner`.
fn probe(img: &image::RgbaImage, probe: &str) -> [u8; 3] {
    if probe == "corner" {
        let p = img.get_pixel(2, 2);
        return [p[0], p[1], p[2]];
    }
    let (cx, cy) = (img.width() / 2 - 2, img.height() / 2 - 2);
    let mut sum = [0u32; 3];
    for y in 0..5 {
        for x in 0..5 {
            let p = img.get_pixel(cx + x, cy + y);
            for c in 0..3 {
                sum[c] += u32::from(p[c]);
            }
        }
    }
    sum.map(|s| ((s as f32) / 25.0).round() as u8)
}

fn rgb(v: &Value) -> [u8; 3] {
    let a = v.as_array().map(Vec::as_slice).unwrap_or_default();
    let c = |i: usize| a.get(i).and_then(Value::as_u64).unwrap_or(0).min(255) as u8;
    [c(0), c(1), c(2)]
}

/// Run every row of `dir/expected.json` on `gpu`.
pub fn run(gpu: &Gpu, root: &Path, dir: &Path) -> anyhow::Result<Receipt> {
    let expected: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json"))?)?;
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("validators/MANIFEST.json"))?)?;
    let sources = &expected["sources"];
    let validator = &manifest["validator"];
    let validator_name = format!(
        "{}@{}",
        validator["tool"].as_str().unwrap_or("?"),
        validator["commit"].as_str().unwrap_or("?").get(..7).unwrap_or("?")
    );
    let mut rows = Vec::new();
    let mut json_rows = Vec::new();
    for row in expected["rows"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let id = row["id"].as_str().unwrap_or("?").to_string();
        let scene_rel = row["scene"].as_str().unwrap_or_default();
        let scene_path: PathBuf = root.join(scene_rel);
        let source_key = row["source"].as_str().unwrap_or_default();
        let source = format!(
            "{} #{}",
            sources[source_key]["short"].as_str().unwrap_or(source_key),
            row["source_row"].as_str().unwrap_or("?")
        );
        let exp = rgb(&row["expected"]);
        let max_delta = row["max_delta"].as_u64().unwrap_or(0).min(255) as u8;
        let probe_name = row["probe"].as_str().unwrap_or("centre5x5").to_string();
        let validation = &manifest["files"][scene_rel];
        let sha = sha256(&scene_path).unwrap_or_default();
        let (mut verdict, mut detail) = (String::new(), String::new());
        if validation.is_null() {
            verdict = "not validated".into();
            detail = format!("{scene_rel} is not in validators/MANIFEST.json");
        } else if validation["sha256"].as_str() != Some(sha.as_str()) {
            verdict = "stale validation".into();
            detail = format!("{scene_rel} changed since {validator_name} validated it: run conformance/validate.sh");
        } else if validation["xsd_valid"] != json!(true) || validation["semantic_errors"] != json!(0) {
            verdict = "invalid scene".into();
            detail = format!("{validator_name}: {validation}");
        }
        let mut got_gamma22 = None;
        let (got, delta) = if verdict.is_empty() {
            let scene = x3d_scene::load(&scene_path)?;
            let size = row["size"].as_array().map_or((128, 128), |s| {
                (
                    s.first().and_then(Value::as_u64).unwrap_or(128) as u32,
                    s.get(1).and_then(Value::as_u64).unwrap_or(128) as u32,
                )
            });
            let shot = Shot { size, msaa: 1, settings: RenderSettings::default(), viewpoint: None };
            let img = render_scene(gpu, &scene, &shot)?;
            let got = probe(&img, &probe_name);
            let g22 = Shot { settings: RenderSettings { encode: Encode::Gamma22, ..Default::default() }, ..shot };
            got_gamma22 = Some(probe(&render_scene(gpu, &scene, &g22)?, &probe_name));
            let delta = (0..3).map(|c| got[c].abs_diff(exp[c])).max().unwrap_or(0);
            verdict = if delta <= max_delta { "pass" } else { "fail" }.into();
            (Some(got), Some(delta))
        } else {
            (None, None)
        };
        json_rows.push(json!({
            "id": id, "scene": scene_rel, "scene_sha256": sha, "probe": probe_name,
            "expected": exp, "expected_from": {"source": sources[source_key], "row": row["source_row"]},
            "got": got, "got_gamma22": got_gamma22, "got_from": describe(&gpu.adapter.get_info()),
            "abs_delta_max": delta, "max_delta": max_delta, "why_tolerance": row["why_tolerance"],
            "validated_by": {"validator": validator, "verdict": validation},
            "verdict": verdict, "detail": detail,
        }));
        rows.push(RowResult {
            id,
            probe: probe_name,
            expected: exp,
            got,
            max_delta,
            delta,
            got_gamma22,
            verdict,
            source,
            detail,
        });
    }
    for row in expected["not_run"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let id = row["id"].as_str().unwrap_or("?").to_string();
        let source_key = row["source"].as_str().unwrap_or_default();
        let reason = row["reason"].as_str().unwrap_or_default().to_string();
        json_rows.push(json!({"id": id, "expected": row["expected"], "expected_from": {"source": sources[source_key], "row": row["source_row"]}, "verdict": "not run", "reason": reason}));
        rows.push(RowResult {
            id,
            probe: "centre5x5".into(),
            expected: rgb(&row["expected"]),
            got: None,
            max_delta: 0,
            delta: None,
            got_gamma22: None,
            verdict: "not run".into(),
            source: format!(
                "{} #{}",
                sources[source_key]["short"].as_str().unwrap_or(source_key),
                row["source_row"].as_str().unwrap_or("?")
            ),
            detail: reason,
        });
    }
    let info = gpu.adapter.get_info();
    let count = |v: &str| rows.iter().filter(|r| r.verdict == v).count();
    let json = json!({
        "receipt": "x3d-wgpu-conformance/1",
        "renderer": {"tool": "x3d-view", "version": env!("CARGO_PKG_VERSION")},
        "adapter": {"name": info.name, "backend": format!("{:?}", info.backend), "device_type": format!("{:?}", info.device_type),
                    "driver": info.driver, "driver_info": info.driver_info},
        "validator": manifest["validator"], "witness_kit": manifest["witness_kit"],
        "summary": {"pass": count("pass"), "fail": count("fail"), "not_run": count("not run"),
                    "blocked": rows.len() - count("pass") - count("fail") - count("not run")},
        "rows": json_rows,
    });
    Ok(Receipt { json, rows })
}
