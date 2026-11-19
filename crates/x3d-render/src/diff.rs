//! Image comparison: the metric behind golden tests and conformance receipts.
//!
//! 🎨 Two renders of the same scene rarely match bit for bit across GPUs:
//! MSAA sample positions, texture filtering and float rounding differ by
//! driver (the Loom measured 258 of 3.9M bytes differing, max 14, between
//! two LLVM versions of the same lavapipe). So we compare with a *named*
//! tolerance: a pixel is "bad" when any channel differs by more than
//! `max_delta`, and an image passes while bad pixels stay under
//! `max_bad_permille` per mille.

use image::{Rgba, RgbaImage};

/// How two images differ (RGB only; alpha is ignored).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diff {
    pub width: u32,
    pub height: u32,
    /// Mean |Δ| over all RGB channels, 0–255 (the x3d.lab metric).
    pub mean_abs: f64,
    /// Largest |Δ| of any channel.
    pub max_abs: u8,
    /// Pixels where some channel differs by more than the threshold.
    pub bad_pixels: u64,
}

impl Diff {
    pub fn bad_permille(&self) -> f64 {
        1000.0 * self.bad_pixels as f64 / (u64::from(self.width) * u64::from(self.height)).max(1) as f64
    }
}

/// A tolerance with the reason it exists.
#[derive(Debug, Clone, PartialEq)]
pub struct Tolerance {
    /// Matched (case-insensitively) against the adapter's name + driver; `*` matches all.
    pub adapter: String,
    pub max_delta: u8,
    pub max_bad_permille: f64,
    pub why: String,
}

impl Tolerance {
    pub fn accepts(&self, d: &Diff) -> bool {
        d.bad_permille() <= self.max_bad_permille
    }

    /// Parse `tests/golden/TOLERANCE`: `adapter-substring  max_delta  max_bad_permille  why…`.
    /// Lines starting with `#` are comments. The first matching line wins.
    pub fn parse_table(text: &str) -> Vec<Tolerance> {
        text.lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let mut it = l.split_whitespace();
                let adapter = it.next()?.to_string();
                let max_delta = it.next()?.parse().ok()?;
                let max_bad_permille = it.next()?.parse().ok()?;
                Some(Tolerance { adapter, max_delta, max_bad_permille, why: it.collect::<Vec<_>>().join(" ") })
            })
            .collect()
    }

    pub fn pick<'a>(table: &'a [Tolerance], adapter: &str) -> Option<&'a Tolerance> {
        let a = adapter.to_lowercase();
        table.iter().find(|t| t.adapter == "*" || a.contains(&t.adapter.to_lowercase()))
    }
}

/// Compare two same-sized images. `None` if the sizes differ.
pub fn diff(a: &RgbaImage, b: &RgbaImage, max_delta: u8) -> Option<Diff> {
    if a.dimensions() != b.dimensions() {
        return None;
    }
    let (mut sum, mut max, mut bad) = (0u64, 0u8, 0u64);
    for (p, q) in a.pixels().zip(b.pixels()) {
        let d = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap_or(0);
        sum += (0..3).map(|c| u64::from(p[c].abs_diff(q[c]))).sum::<u64>();
        max = max.max(d);
        bad += u64::from(d > max_delta);
    }
    let (width, height) = a.dimensions();
    let n = (u64::from(width) * u64::from(height) * 3).max(1);
    Some(Diff { width, height, mean_abs: sum as f64 / n as f64, max_abs: max, bad_pixels: bad })
}

/// A heat map: black where equal, brighter red the bigger the difference,
/// yellow where a pixel is over `max_delta`.
pub fn heatmap(a: &RgbaImage, b: &RgbaImage, max_delta: u8) -> RgbaImage {
    RgbaImage::from_fn(a.width().min(b.width()), a.height().min(b.height()), |x, y| {
        let (p, q) = (a.get_pixel(x, y), b.get_pixel(x, y));
        let d = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap_or(0);
        if d > max_delta { Rgba([255, 255, 0, 255]) } else { Rgba([d.saturating_mul(8), 0, 0, 255]) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_counts_and_tolerates() {
        let a = RgbaImage::from_pixel(10, 10, Rgba([100, 100, 100, 255]));
        let mut b = a.clone();
        b.put_pixel(0, 0, Rgba([110, 100, 100, 255]));
        b.put_pixel(1, 0, Rgba([102, 100, 100, 255]));
        let d = diff(&a, &b, 4).unwrap();
        assert_eq!((d.max_abs, d.bad_pixels), (10, 1));
        assert!((d.bad_permille() - 10.0).abs() < 1e-9);
        let t = Tolerance::parse_table("# c\nllvmpipe 2 0 exact-ish\n* 8 12.5 any other GPU\n");
        assert_eq!(Tolerance::pick(&t, "llvmpipe (LLVM 20)").unwrap().max_delta, 2);
        assert_eq!(Tolerance::pick(&t, "Intel UHD 620").unwrap().why, "any other GPU");
        assert!(diff(&a, &RgbaImage::new(3, 3), 0).is_none());
    }
}
