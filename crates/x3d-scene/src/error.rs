//! Errors for loading.
//!
//! 🦀 Library crates expose a *typed* error (this enum) so callers can
//! `match` on what went wrong. Binaries usually wrap it in `anyhow::Error`
//! and add context. `thiserror` writes the `Display` and `Error` impls for us.

use std::path::PathBuf;

use crate::fields::FieldError;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive] // 🦀 we may add variants later without breaking callers
pub enum Error {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{0} is not UTF-8 text")]
    NotUtf8(PathBuf),

    /// 🦀 `#[from]` generates `impl From<roxmltree::Error> for Error`,
    /// which is what lets `?` convert the error automatically.
    #[error("XML: {0}")]
    Xml(#[from] roxmltree::Error),

    #[error("no <Scene> element (is this an X3D XML file?)")]
    NoScene,

    #[error("<{node}> on line {line}: field `{field}`: {source}")]
    Field {
        node: String,
        field: String,
        line: u32,
        #[source]
        source: FieldError,
    },

    #[error("<{node}> on line {line}: {message}")]
    Invalid { node: String, line: u32, message: String },

    #[error("USE='{0}' has no matching DEF")]
    MissingDef(String),

    #[error("nodes nested deeper than {0} levels (a USE cycle?)")]
    TooDeep(usize),
}

/// 🦀 A crate-wide alias so signatures stay short: `Result<Scene>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;
