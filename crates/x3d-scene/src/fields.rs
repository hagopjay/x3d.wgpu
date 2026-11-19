//! X3D *field* values: the text inside attributes like `size='2 2 2'`.
//!
//! X3D (XML encoding) stores every value as a string. Single-value fields
//! are `SF…` (SFVec3f = one 3-vector) and multi-value fields are `MF…`
//! (MFVec3f = a list of 3-vectors). Numbers are separated by whitespace
//! and/or commas, so `"0 0 0, 1 0 0"` and `"0,0,0 1,0,0"` are both fine.
//!
//! 🦀 **Rust idioms in this file**
//! - A trait ([`FieldValue`]) so one generic function can parse *any*
//!   field type: `node.field::<Vec3>("size")`.
//! - Const generics (`[f32; N]`) so the same code parses 2-, 3- and
//!   4-vectors without copy-paste.
//! - Iterators that borrow the input (`impl Iterator + '_`): no
//!   allocation until we `collect()`.
//! - `thiserror` for a typed error enum that still prints nicely.

use glam::{Quat, Vec2, Vec3, Vec4};

/// Why a field's text could not be parsed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FieldError {
    #[error("`{0}` is not a number")]
    BadNumber(String),
    #[error("`{0}` is not a boolean (expected true/false)")]
    BadBool(String),
    #[error("expected {expected} numbers, found {found}")]
    WrongCount { expected: usize, found: usize },
    #[error("{found} numbers is not a multiple of {group}")]
    NotAMultiple { group: usize, found: usize },
}

/// Split a field string into number tokens.
///
/// ```
/// use x3d_scene::fields::tokens;
/// let t: Vec<&str> = tokens("1, 2  3,4").collect();
/// assert_eq!(t, ["1", "2", "3", "4"]);
/// ```
pub fn tokens(s: &str) -> impl Iterator<Item = &str> + '_ {
    // 🦀 `split` takes any `Pattern`; a closure works. `filter` drops the
    // empty strings produced by ", " (comma followed by a space).
    s.split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|t| !t.is_empty())
}

/// Parse every token as `f32`.
pub fn floats(s: &str) -> impl Iterator<Item = Result<f32, FieldError>> + '_ {
    tokens(s).map(|t| {
        t.parse::<f32>()
            .map_err(|_| FieldError::BadNumber(t.to_owned()))
    })
}

/// Parse every token as `i32` (for index lists like `coordIndex`).
pub fn ints(s: &str) -> Result<Vec<i32>, FieldError> {
    // 🦀 Collecting an iterator of `Result<T, E>` into `Result<Vec<T>, E>`
    // stops at the first error. One of the most useful tricks in Rust.
    tokens(s)
        .map(|t| {
            t.parse::<i32>()
                .map_err(|_| FieldError::BadNumber(t.to_owned()))
        })
        .collect()
}

/// Exactly `N` floats, e.g. an SFVec3f.
///
/// ```
/// use x3d_scene::fields::fixed;
/// assert_eq!(fixed::<3>("1 2 3").unwrap(), [1.0, 2.0, 3.0]);
/// assert!(fixed::<3>("1 2").is_err());
/// ```
pub fn fixed<const N: usize>(s: &str) -> Result<[f32; N], FieldError> {
    let v: Vec<f32> = floats(s).collect::<Result<_, _>>()?;
    // 🦀 `TryFrom<Vec<T>> for [T; N]` gives the array back if the length is
    // right, or the Vec back (in `Err`) if it isn't.
    <[f32; N]>::try_from(v).map_err(|v| FieldError::WrongCount {
        expected: N,
        found: v.len(),
    })
}

/// Groups of `N` floats, e.g. an MFVec3f. A trailing partial group is an error.
///
/// ```
/// use x3d_scene::fields::grouped;
/// let pts = grouped::<3>("0 0 0, 1 0 0").unwrap();
/// assert_eq!(pts.len(), 2);
/// ```
pub fn grouped<const N: usize>(s: &str) -> Result<Vec<[f32; N]>, FieldError> {
    let v: Vec<f32> = floats(s).collect::<Result<_, _>>()?;
    let chunks = v.chunks_exact(N);
    if !chunks.remainder().is_empty() {
        return Err(FieldError::NotAMultiple {
            group: N,
            found: v.len(),
        });
    }
    // `chunk` is `&[f32]` with len N, so `try_into` cannot fail here.
    Ok(chunks
        .map(|chunk| chunk.try_into().expect("chunks_exact"))
        .collect())
}

/// A type that can be parsed from an X3D attribute string.
///
/// Implement this for a new type and every `field::<YourType>()` call
/// works automatically. That is the power of traits + generics.
pub trait FieldValue: Sized {
    fn parse_field(s: &str) -> Result<Self, FieldError>;
}

impl FieldValue for f32 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        fixed::<1>(s).map(|[x]| x) // 🦀 destructure a 1-element array
    }
}

impl FieldValue for i32 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        let v = ints(s)?;
        match v.as_slice() {
            [x] => Ok(*x),
            _ => Err(FieldError::WrongCount {
                expected: 1,
                found: v.len(),
            }),
        }
    }
}

impl FieldValue for bool {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        // X3D XML says `true`/`false`; VRML-era files often say TRUE/FALSE.
        match s.trim() {
            "true" | "TRUE" => Ok(true),
            "false" | "FALSE" => Ok(false),
            other => Err(FieldError::BadBool(other.to_owned())),
        }
    }
}

impl FieldValue for Vec2 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        fixed::<2>(s).map(Vec2::from)
    }
}

impl FieldValue for Vec3 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        fixed::<3>(s).map(Vec3::from)
    }
}

impl FieldValue for Vec4 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        fixed::<4>(s).map(Vec4::from)
    }
}

/// SFRotation is `axisX axisY axisZ angle` (radians), stored as a quaternion.
impl FieldValue for Quat {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        let [x, y, z, angle] = fixed::<4>(s)?;
        Ok(rotation(Vec3::new(x, y, z), angle))
    }
}

/// Axis-angle → quaternion, tolerating a zero axis (common in exported files).
pub fn rotation(axis: Vec3, angle: f32) -> Quat {
    // 🦀 `try_normalize` returns `Option<Vec3>`: `None` for a zero-length axis.
    axis.try_normalize()
        .map_or(Quat::IDENTITY, |a| Quat::from_axis_angle(a, angle))
}

impl FieldValue for Vec<f32> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        floats(s).collect()
    }
}

impl FieldValue for Vec<i32> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        ints(s)
    }
}

impl FieldValue for Vec<Vec2> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        Ok(grouped::<2>(s)?.into_iter().map(Vec2::from).collect())
    }
}

impl FieldValue for Vec<Vec3> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        Ok(grouped::<3>(s)?.into_iter().map(Vec3::from).collect())
    }
}

impl FieldValue for Vec<Vec4> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        Ok(grouped::<4>(s)?.into_iter().map(Vec4::from).collect())
    }
}

impl FieldValue for f64 {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        s.trim()
            .parse()
            .map_err(|_| FieldError::BadNumber(s.trim().to_owned()))
    }
}

/// MFRotation: groups of four (axis x y z, angle).
impl FieldValue for Vec<Quat> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        Ok(grouped::<4>(s)?
            .into_iter()
            .map(|[x, y, z, a]| rotation(Vec3::new(x, y, z), a))
            .collect())
    }
}

/// MFBool: `true false TRUE …`.
impl FieldValue for Vec<bool> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        tokens(s).map(bool::parse_field).collect()
    }
}

/// MFString: `'"EXAMINE" "ANY"'` → `["EXAMINE", "ANY"]`.
/// Good enough for the fields we read; escaped quotes are not handled.
impl FieldValue for Vec<String> {
    fn parse_field(s: &str) -> Result<Self, FieldError> {
        if !s.contains('"') {
            return Ok(vec![s.trim().to_owned()]);
        }
        Ok(s.split('"')
            .skip(1)
            .step_by(2) // every second piece is inside quotes
            .map(str::to_owned)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commas_and_whitespace_both_separate() {
        let v: Vec<Vec3> = FieldValue::parse_field("0 0 0, 1,0,0\n2 0 0").unwrap();
        assert_eq!(v, [Vec3::ZERO, Vec3::X, Vec3::X * 2.0]);
    }

    #[test]
    fn errors_are_descriptive() {
        let e = f32::parse_field("abc").unwrap_err();
        assert_eq!(e.to_string(), "`abc` is not a number");
        let e = grouped::<3>("1 2 3 4").unwrap_err();
        assert_eq!(e, FieldError::NotAMultiple { group: 3, found: 4 });
    }

    #[test]
    fn booleans_accept_vrml_case() {
        assert!(bool::parse_field("TRUE").unwrap());
        assert!(!bool::parse_field(" false ").unwrap());
        assert!(bool::parse_field("yes").is_err());
    }

    #[test]
    fn zero_axis_rotation_is_identity() {
        assert_eq!(Quat::parse_field("0 0 0 1.57").unwrap(), Quat::IDENTITY);
    }

    #[test]
    fn mfstring() {
        let v: Vec<String> = FieldValue::parse_field(r#""EXAMINE" "ANY""#).unwrap();
        assert_eq!(v, ["EXAMINE", "ANY"]);
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(f32::parse_field("1e-3").unwrap(), 0.001);
    }
}
