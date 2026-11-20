//! Vulkan Validation Layer (VVL) evidence for receipts.
//!
//! 🎨 The Khronos validation layer (`VK_LAYER_KHRONOS_validation`) sits between
//! the renderer and the Vulkan driver and checks every call against the spec.
//! A clean run on lavapipe says "this renderer speaks legal Vulkan", which is
//! what lets a receipt be trusted on drivers that forgive illegal calls.
//!
//! wgpu's Vulkan backend turns the layer on with `InstanceFlags::VALIDATION`
//! and forwards each layer message to the `log` crate (target
//! `wgpu_hal::vulkan::instance`, format `"{type} [{VUID name} (0x{number})]\n\t{text}"`,
//! level = the layer's severity). This module installs a logger that *keeps*
//! the warnings and errors, so a receipt can count them per row.
//!
//! `X3D_VVL` selects the mode: unset/`0` leave wgpu's default (on in debug
//! builds only), `1` asks for the layer, `require` asks and treats a missing
//! layer or any error as a failure (CI uses this).
//!
//! **Honesty notes.** wgpu-hal itself silences a few VUIDs
//! ([`SUPPRESSED_BY_WGPU_HAL`]); the layer's version is read from its JSON
//! manifest, because wgpu does not expose it; and messages are attributed to
//! a row by time, so run the checks serially (CI does).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub const LAYER: &str = "VK_LAYER_KHRONOS_validation";
const TARGET: &str = "wgpu_hal::vulkan::instance";

/// VUIDs that wgpu-hal 30 drops before logging (see its
/// `debug_utils_messenger_callback`). A receipt lists them so a clean count
/// means "clean, except for these".
pub const SUPPRESSED_BY_WGPU_HAL: &[(&str, &str)] = &[
    ("VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912", "layers 1.3.240-1.3.250 only (spurious)"),
    ("VUID-VkSwapchainCreateInfoKHR-pNext-07781", "surface resize race; not used by headless runs"),
    ("VUID-VkRenderPassBeginInfo-framebuffer-04627", "only with the OBS layer"),
    ("VUID-vkCmdCopyImageToBuffer-pRegions-00184", "suspected layer bug, KhronosGroup/Vulkan-ValidationLayers#9276"),
    ("VUID-StandaloneSpirv-None-10684", "wgpu issue 7696"),
];

/// Layer *errors* this project knows about and tracks (docs/VVL.md). They are
/// counted and listed in every receipt but do not fail a run; any other error
/// does. Keep this list short and each entry honest.
pub const KNOWN_ERRORS: &[(&str, &str)] = &[(
    "SYNC-HAZARD-READ-AFTER-WRITE",
    "wide lines (scenes/04, render tests lines_have_their_x3d_width, a_line_running_behind_the_camera): vertex-attribute read after a staging copy in wgpu's PendingWrites, no barrier. Cause not found; see docs/VVL.md",
)];

fn is_known(id: &str) -> bool {
    KNOWN_ERRORS.iter().any(|(k, _)| *k == id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// wgpu's default (the layer is on in debug builds only).
    Default,
    /// Ask for the layer.
    On,
    /// Ask for it; a missing layer or any error fails the run.
    Require,
}

pub fn mode() -> Mode {
    match std::env::var("X3D_VVL").ok().as_deref().map(str::trim) {
        Some("1" | "on" | "true") => Mode::On,
        Some("require" | "required") => Mode::Require,
        _ => Mode::Default,
    }
}

/// One warning or error from the layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    /// "error" or "warning".
    pub severity: &'static str,
    /// The VUID name, e.g. `VUID-vkCmdDraw-None-02699` (empty if absent).
    pub id: String,
    /// The numeric message id, hex, as the layer reports it.
    pub number: String,
    pub text: String,
}

static MESSAGES: Mutex<Vec<Message>> = Mutex::new(Vec::new());
static LAYER_MISSING: AtomicBool = AtomicBool::new(false);
static FLAG_ASKED: AtomicBool = AtomicBool::new(false);
static INSTANCES: Mutex<u32> = Mutex::new(0);

/// Keeps VVL warnings/errors, forwards everything to env_logger.
struct Capture(env_logger::Logger);

impl log::Log for Capture {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target() == TARGET && m.level() <= log::Level::Warn || self.0.enabled(m)
    }

    fn log(&self, r: &log::Record) {
        if r.target() == TARGET {
            let text = r.args().to_string();
            if text.contains("unable to find layer") {
                LAYER_MISSING.store(true, Ordering::SeqCst);
            } else if let Some(m) = parse(r.level(), &text) {
                if let Ok(mut v) = MESSAGES.lock() {
                    v.push(m);
                }
            }
        }
        if self.0.matches(r) {
            self.0.log(r);
        }
    }

    fn flush(&self) {
        self.0.flush();
    }
}

/// `"VALIDATION [VUID-x (0x1a2b)]\n\tmessage"` → a [`Message`].
fn parse(level: log::Level, text: &str) -> Option<Message> {
    let severity = match level {
        log::Level::Error => "error",
        log::Level::Warn => "warning",
        _ => return None,
    };
    let (head, body) = text.split_once("\n\t").unwrap_or((text, ""));
    // The layer's extra lines ("objects: …", queue labels) are logged as separate
    // records without a message id: they are not messages of their own.
    let (id, number) = match (head.find('['), head.rfind(" (0x")) {
        (Some(a), Some(b)) if b > a => (head[a + 1..b].to_string(), head[b + 4..].trim_end_matches(")]").to_string()),
        _ => return None,
    };
    Some(Message { severity, id, number, text: body.chars().take(400).collect() })
}

/// Install the capturing logger (once). Call before creating any [`crate::Gpu`].
/// `default_filter` is env_logger's fallback when `RUST_LOG` is unset.
pub fn install(default_filter: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let inner = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_filter)).build();
        // Warn is the floor: we must see the layer's warnings even under `RUST_LOG=error`.
        let max = inner.filter().max(log::LevelFilter::Warn);
        if log::set_boxed_logger(Box::new(Capture(inner))).is_ok() {
            log::set_max_level(max);
        }
    });
}

/// Called by `gpu::make_instance`: were validation flags requested?
pub(crate) fn note_instance(flags_have_validation: bool) {
    FLAG_ASKED.store(flags_have_validation, Ordering::SeqCst);
    if let Ok(mut n) = INSTANCES.lock() {
        *n += 1;
    }
}

/// Position in the message list, to attribute later messages to one row.
pub fn mark() -> usize {
    MESSAGES.lock().map_or(0, |v| v.len())
}

pub fn since(mark: usize) -> Vec<Message> {
    MESSAGES.lock().map(|v| v.iter().skip(mark).cloned().collect()).unwrap_or_default()
}

/// Is the layer really running? (Asked for, and not reported missing.)
pub fn enabled() -> bool {
    FLAG_ASKED.load(Ordering::SeqCst) && !LAYER_MISSING.load(Ordering::SeqCst)
}

/// What we know about the installed layer, from its JSON manifest.
#[derive(Debug, Clone, Default)]
pub struct LayerInfo {
    pub version: Option<String>,
    pub source: Option<PathBuf>,
}

pub fn layer_info() -> LayerInfo {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for var in ["VK_LAYER_PATH", "VK_ADD_LAYER_PATH"] {
        if let Some(v) = std::env::var_os(var) {
            dirs.extend(std::env::split_paths(&v));
        }
    }
    for d in [
        "/usr/share/vulkan/explicit_layer.d",
        "/etc/vulkan/explicit_layer.d",
        "/usr/local/share/vulkan/explicit_layer.d",
    ] {
        dirs.push(d.into());
    }
    for dir in dirs {
        let path = dir.join("VkLayer_khronos_validation.json");
        if let Some(info) = read_manifest(&path) {
            return info;
        }
    }
    LayerInfo::default()
}

fn read_manifest(path: &Path) -> Option<LayerInfo> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let layer = &v["layer"];
    if layer["name"].as_str()? != LAYER {
        return None;
    }
    let version = layer["api_version"]
        .as_str()
        .map(|a| format!("{a} (implementation_version {})", layer["implementation_version"].as_str().unwrap_or("?")));
    Some(LayerInfo { version, source: Some(path.to_path_buf()) })
}

/// Totals for a set of messages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    pub errors: usize,
    pub warnings: usize,
    /// Errors whose id is not in [`KNOWN_ERRORS`]: these fail a run.
    pub unexpected_errors: usize,
}

pub fn counts(msgs: &[Message]) -> Counts {
    Counts {
        errors: msgs.iter().filter(|m| m.severity == "error").count(),
        warnings: msgs.iter().filter(|m| m.severity == "warning").count(),
        unexpected_errors: msgs.iter().filter(|m| m.severity == "error" && !is_known(&m.id)).count(),
    }
}

/// Distinct VUID names, sorted.
pub fn message_ids(msgs: &[Message]) -> Vec<String> {
    let mut ids: Vec<String> = msgs.iter().map(|m| m.id.clone()).collect();
    ids.sort();
    ids.dedup();
    ids
}

/// The per-row block: just counts and ids.
pub fn row_block(msgs: &[Message]) -> Value {
    let c = counts(msgs);
    json!({"errors": c.errors, "unexpected_errors": c.unexpected_errors, "warnings": c.warnings, "message_ids": message_ids(msgs)})
}

/// The receipt-level `api_validation` block (shared schema: see
/// `docs/RECEIPT-SCHEMA.md`). `all` is every message since the run began.
pub fn receipt_block(all: &[Message]) -> Value {
    let info = layer_info();
    let c = counts(all);
    let mut grouped: Vec<(String, String, &'static str, usize, String)> = Vec::new();
    for m in all {
        if let Some(g) = grouped.iter_mut().find(|g| g.0 == m.id && g.2 == m.severity) {
            g.3 += 1;
        } else {
            grouped.push((m.id.clone(), m.number.clone(), m.severity, 1, m.text.clone()));
        }
    }
    let log_text: String = all
        .iter()
        .map(|m| format!("{} {} (0x{}) {}", m.severity, m.id, m.number, m.text))
        .collect::<Vec<_>>()
        .join("\n");
    let sha: String = Sha256::digest(log_text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    json!({
        "layer": if enabled() || mode() != Mode::Default { json!(LAYER) } else { Value::Null },
        "layer_version": info.version,
        "layer_version_source": info.source.map(|p| p.display().to_string()),
        "requested": FLAG_ASKED.load(Ordering::SeqCst),
        "mode": format!("{:?}", mode()),
        "enabled": enabled(),
        "errors": c.errors,
        "unexpected_errors": c.unexpected_errors,
        "warnings": c.warnings,
        "message_ids": message_ids(all),
        "known_errors": KNOWN_ERRORS.iter().map(|(id, why)| json!({"id": id, "why": why})).collect::<Vec<_>>(),
        "messages": grouped.iter().map(|(id, number, severity, count, text)| json!({
            "severity": severity, "id": id, "number": number, "count": count, "first_text": text})).collect::<Vec<_>>(),
        "suppressed_by_engine": SUPPRESSED_BY_WGPU_HAL.iter().map(|(id, why)| json!({"id": id, "why": why})).collect::<Vec<_>>(),
        "log_sha256": sha,
        "attribution": "messages are attributed to a row by time; run serially",
    })
}

/// Does `Require` mode accept this run? (`Ok` when the mode is not `Require`.)
pub fn check_required(all: &[Message]) -> Result<(), String> {
    check(all, mode(), enabled())
}

fn check(all: &[Message], mode: Mode, enabled: bool) -> Result<(), String> {
    if mode != Mode::Require {
        return Ok(());
    }
    if !enabled {
        return Err(format!(
            "X3D_VVL=require but {LAYER} is not running (not installed? `apt install vulkan-validationlayers`)"
        ));
    }
    let c = counts(all);
    if c.unexpected_errors > 0 {
        return Err(format!(
            "{} unexpected Vulkan validation error(s): {}",
            c.unexpected_errors,
            message_ids(all).join(", ")
        ));
    }
    Ok(())
}

/// For tests: panic if the layer reported an error since the process began
/// (or, in `Require` mode, if the layer is not running).
pub fn assert_clean(who: &str) {
    let all = since(0);
    if let Err(why) = check_required(&all) {
        panic!("{who}: {why}");
    }
    let c = counts(&all);
    assert!(
        !(enabled() && c.unexpected_errors > 0),
        "{who}: {} unexpected Vulkan validation error(s): {}",
        c.unexpected_errors,
        message_ids(&all).join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_wgpu_hal_log_format() {
        let m = parse(log::Level::Error, "VALIDATION [VUID-vkCmdDraw-None-02699 (0x1a2b3c)]\n\tbad things").unwrap();
        assert_eq!((m.severity, m.id.as_str(), m.number.as_str()), ("error", "VUID-vkCmdDraw-None-02699", "1a2b3c"));
        assert_eq!(m.text, "bad things");
        assert!(parse(log::Level::Debug, "x").is_none());
        assert!(parse(log::Level::Error, "\tobjects: (type: QUEUE, hndl: 0x1, name: ?)").is_none());
        let w = parse(log::Level::Warn, "PERFORMANCE [UNASSIGNED-x (0x0)]\n\tslow").unwrap();
        assert_eq!(w.severity, "warning");
    }

    #[test]
    fn counts_and_ids() {
        let m =
            |s: &'static str, id: &str| Message { severity: s, id: id.into(), number: "0".into(), text: String::new() };
        let all = [m("error", "B"), m("warning", "A"), m("error", "B")];
        assert_eq!(counts(&all), Counts { errors: 2, warnings: 1, unexpected_errors: 2 });
        let known = [m("error", "SYNC-HAZARD-READ-AFTER-WRITE")];
        assert_eq!(counts(&known).unexpected_errors, 0);
        assert_eq!(message_ids(&all), ["A", "B"]);
    }

    #[test]
    fn the_gate_fails_on_unexpected_errors_and_on_a_missing_layer() {
        let m =
            |s: &'static str, id: &str| Message { severity: s, id: id.into(), number: "0".into(), text: String::new() };
        // A new error id fails in require mode…
        let bad = [m("error", "VUID-vkCmdDraw-None-02699")];
        assert!(check(&bad, Mode::Require, true).unwrap_err().contains("VUID-vkCmdDraw-None-02699"));
        // …a known one and a warning do not…
        assert!(check(&[m("error", "SYNC-HAZARD-READ-AFTER-WRITE"), m("warning", "X")], Mode::Require, true).is_ok());
        // …a missing layer fails even with no messages…
        assert!(check(&[], Mode::Require, false).unwrap_err().contains("not running"));
        // …and without require nothing fails here.
        assert!(check(&bad, Mode::On, true).is_ok());
    }

    #[test]
    fn reads_a_layer_manifest() {
        let dir = std::env::temp_dir().join(format!("x3d-vvl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("VkLayer_khronos_validation.json");
        std::fs::write(&p, r#"{"layer": {"name": "VK_LAYER_KHRONOS_validation", "api_version": "1.4.321", "implementation_version": "1"}}"#)
            .unwrap();
        let info = read_manifest(&p).unwrap();
        assert_eq!(info.version.as_deref(), Some("1.4.321 (implementation_version 1)"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
