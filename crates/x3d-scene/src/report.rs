//! An honest account of what loading kept, approximated and dropped.
//!
//! The same idea as x3d.lab's `EXPECTED_LOSS`: a renderer that silently
//! drops nodes is hard to trust. Every node we skip is counted here, and
//! `x3d-view info scene.x3d` prints it.

use std::collections::BTreeMap;
use std::fmt;

/// 🦀 `BTreeMap` (not `HashMap`) so the printout is sorted and stable.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// X3D node name → how many were skipped.
    pub unsupported: BTreeMap<String, usize>,
    /// Supported, but not exactly as X3D says (e.g. Billboard as Group).
    pub approximated: BTreeMap<String, usize>,
    /// Animation and interaction nodes: this renderer draws a still.
    pub ignored: BTreeMap<String, usize>,
    /// Problems with specific nodes, e.g. a bad field value.
    pub warnings: Vec<String>,
    pub stats: Stats,
    /// Files pulled in through Inline (nested ones too), in load order.
    pub inlined: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub shapes: usize,
    pub unique_meshes: usize,
    pub triangles: usize,
    pub line_segments: usize,
    pub points: usize,
    pub lights: usize,
    pub viewpoints: usize,
    /// ROUTEs wired into the event graph.
    pub routes: usize,
}

impl Report {
    pub fn unsupported(&mut self, node: &str) {
        *self.unsupported.entry(node.to_owned()).or_default() += 1;
    }
    pub fn approximated(&mut self, what: &str) {
        *self.approximated.entry(what.to_owned()).or_default() += 1;
    }
    pub fn ignored(&mut self, node: &str) {
        *self.ignored.entry(node.to_owned()).or_default() += 1;
    }
    pub fn warn(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        log::warn!("{msg}");
        self.warnings.push(msg);
    }
    /// Fold in the report of an inlined file.
    pub fn absorb(&mut self, other: &Report, from: &str) {
        for (mine, theirs) in [
            (&mut self.unsupported, &other.unsupported),
            (&mut self.approximated, &other.approximated),
            (&mut self.ignored, &other.ignored),
        ] {
            for (k, n) in theirs {
                *mine.entry(k.clone()).or_default() += n;
            }
        }
        self.warnings.extend(other.warnings.iter().map(|w| format!("{from}: {w}")));
        self.inlined.push(from.to_owned());
        self.inlined.extend(other.inlined.iter().cloned());
    }

    /// Nothing was skipped or approximated.
    pub fn is_lossless(&self) -> bool {
        self.unsupported.is_empty()
            && self.approximated.is_empty()
            && self.ignored.is_empty()
            && self.warnings.is_empty()
    }
}

/// 🦀 `Display` is for humans; `Debug` (derived above) is for developers.
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.stats;
        writeln!(
            f,
            "shapes {}  meshes {}  triangles {}  line segments {}  points {}  lights {}  viewpoints {}  routes {}",
            s.shapes, s.unique_meshes, s.triangles, s.line_segments, s.points, s.lights, s.viewpoints, s.routes
        )?;
        if !self.inlined.is_empty() {
            writeln!(f, "inlined {} file(s): {}", self.inlined.len(), self.inlined.join(", "))?;
        }
        if self.is_lossless() {
            return writeln!(f, "nothing skipped ✔");
        }
        // A small closure keeps the two tables identical in format.
        let mut table = |title: &str, map: &BTreeMap<String, usize>| -> fmt::Result {
            if map.is_empty() {
                return Ok(());
            }
            writeln!(f, "{title}:")?;
            for (name, n) in map {
                writeln!(f, "  {n:>5} × {name}")?;
            }
            Ok(())
        };
        table("skipped (not supported yet)", &self.unsupported)?;
        table("approximated", &self.approximated)?;
        table("ignored (interaction, and ROUTEs not followed yet)", &self.ignored)?;
        if !self.warnings.is_empty() {
            writeln!(f, "warnings:")?;
            for w in &self.warnings {
                writeln!(f, "  - {w}")?;
            }
        }
        Ok(())
    }
}
