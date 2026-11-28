//! What the loader says it handles, by node name: the static column of the
//! multi-renderer audit. (The dynamic column comes from loading a real scene
//! and reading its `Report`; the two should agree, and the audit compares them.)

/// How a node is treated, by the loader's own match arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Drawn or wired up.
    Drawn,
    /// Drawn, but not exactly as X3D says.
    Approximated,
    /// Loaded without error but not drawn (metadata, prototypes).
    Silent,
    /// Counted in the loss report, not drawn.
    Skipped,
    /// Works only in a particular setting; a lone node of this kind is
    /// reported as skipped (seen by the 260-node audit, 28 Nov).
    Conditional,
}

impl Support {
    pub fn as_str(self) -> &'static str {
        match self {
            Support::Drawn => "drawn",
            Support::Approximated => "approximated",
            Support::Silent => "silent",
            Support::Skipped => "skipped",
            Support::Conditional => "conditional",
        }
    }
}

const DRAWN: &[&str] = &[
    "Transform",
    "Group",
    "StaticGroup",
    "Switch",
    "LOD",
    "Collision",
    "Anchor",
    "Inline",
    "Shape",
    "Appearance",
    "Box",
    "Sphere",
    "Cylinder",
    "Cone",
    "IndexedFaceSet",
    "ElevationGrid",
    "IndexedLineSet",
    "LineSet",
    "PointSet",
    "TriangleSet",
    "IndexedTriangleSet",
    "QuadSet",
    "IndexedQuadSet",
    "TriangleFanSet",
    "IndexedTriangleFanSet",
    "TriangleStripSet",
    "IndexedTriangleStripSet",
    "Material",
    "UnlitMaterial",
    "PhysicalMaterial",
    "LineProperties",
    "PointProperties",
    "ImageTexture",
    "PixelTexture",
    "TextureCoordinate",
    "TextureCoordinate3D",
    "TextureTransform",
    "TextureTransform3D",
    "TextureTransformMatrix3D",
    "Coordinate",
    "Color",
    "ColorRGBA",
    "Normal",
    "DirectionalLight",
    "PointLight",
    "SpotLight",
    "EnvironmentLight",
    "ComposedCubeMapTexture",
    "Viewpoint",
    "NavigationInfo",
    "Background",
    "Fog",
    "TimeSensor",
    "ScalarInterpolator",
    "PositionInterpolator",
    "PositionInterpolator2D",
    "ColorInterpolator",
    "OrientationInterpolator",
    "IntegerSequencer",
    "BooleanSequencer",
    "BooleanFilter",
    "BooleanToggle",
    "ComposedShader",
    "ShaderPart",
    "ROUTE",
];

const APPROXIMATED: &[&str] =
    &["Billboard", "OrthoViewpoint", "MultiTextureCoordinate", "TextureCoordinateGenerator", "TwoSidedMaterial"];

const SILENT: &[&str] =
    &["WorldInfo", "ProtoDeclare", "ExternProtoDeclare", "IMPORT", "EXPORT", "IS", "connect", "field", "fieldValue"];

/// Drawn only when wired up: BooleanToggle and PositionInterpolator2D need a
/// ROUTE; ComposedCubeMapTexture is read only by an EnvironmentLight.
const CONDITIONAL: &[&str] = &["BooleanToggle", "PositionInterpolator2D", "ComposedCubeMapTexture"];

pub fn static_support(name: &str) -> Support {
    if CONDITIONAL.contains(&name) {
        Support::Conditional
    } else if DRAWN.contains(&name)
        || name.starts_with("CAD") && matches!(name, "CADAssembly" | "CADLayer" | "CADPart" | "CADFace")
    {
        Support::Drawn
    } else if APPROXIMATED.contains(&name) {
        Support::Approximated
    } else if SILENT.contains(&name) {
        Support::Silent
    } else {
        Support::Skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(static_support("Box"), Support::Drawn);
        assert_eq!(static_support("Billboard"), Support::Approximated);
        assert_eq!(static_support("WorldInfo"), Support::Silent);
        assert_eq!(static_support("Text"), Support::Skipped);
        assert_eq!(static_support("CADPart"), Support::Drawn);
    }
}
