//! Walking the X3D XML tree.
//!
//! 🦀 **Lifetimes, for real.** `roxmltree` parses without copying: every
//! `Node<'a, 'input>` borrows from the `Document<'input>`, which borrows
//! from the XML `&'input str`. Our [`Loader`] borrows them too, so the whole
//! walk allocates nothing for names or attribute text. The compiler proves
//! none of those borrows outlive the string.
//!
//! 🦀 **Extension traits.** We can't add methods to `roxmltree::Node`
//! (it's someone else's type), but we *can* define our own trait
//! ([`NodeExt`]) and implement it for their type.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use roxmltree::{Document, Node, NodeId};

use crate::error::{Error, Result};
use crate::fields::FieldValue;
use crate::geometry::{self, IndexError, Parts, Polygons};
use crate::runtime::{
    AnimNode, Behaviour, KeyValues, Local, Route, Target, TextureTransform2D, TransformNode, Trs, Value, field_name,
};
use crate::scene::{
    Background, CubeMap, Draw, Environment, Light, LightKind, LodTag, Material, Mesh, Pbr, Scene, ShaderSource,
    SwitchGuard, TextureRef, Textures, Topology, Viewpoint,
};

/// Knobs for loading.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Segments around Sphere/Cylinder/Cone.
    pub segments: u32,
    /// `true`: a bad Shape fails the whole load. `false` (default): it is
    /// skipped and listed in the report's warnings.
    pub strict: bool,
    /// Also load `Inline`s marked `load='false'` (a browser would load them
    /// later, when a script or sensor asks; a still has no "later").
    pub load_all_inlines: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { segments: 48, strict: false, load_all_inlines: false }
    }
}

/// Bookkeeping shared by one whole load, across every Inlined file.
#[derive(Debug, Default)]
struct InlineState {
    /// Files being loaded right now, outermost first: a cycle detector.
    stack: Vec<PathBuf>,
    /// Each file is parsed once, however often it's Inlined.
    /// 🦀 `Arc<Scene>`: shared, immutable, cheap to hand out again.
    cache: HashMap<PathBuf, Arc<Scene>>,
}

const MAX_DEPTH: usize = 256;

/// Load an `.x3d` (or gzipped `.x3dz`) file.
pub fn load(path: impl AsRef<Path>) -> Result<Scene> {
    load_with(path, &LoadOptions::default())
}

/// Load a file with options.
pub fn load_with(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Scene> {
    let path = path.as_ref();
    let path = path.canonicalize().unwrap_or_else(|_| path.to_owned());
    let mut state = InlineState { stack: vec![path.clone()], ..Default::default() };
    load_file(&path, opts, &mut state)
}

fn load_file(path: &Path, opts: &LoadOptions, state: &mut InlineState) -> Result<Scene> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io { path: path.to_owned(), source })?;
    let bytes = maybe_gunzip(bytes, path)?;
    let text = String::from_utf8(bytes).map_err(|_| Error::NotUtf8(path.to_owned()))?;
    parse_inner(&text, opts, path.parent(), state)
}

#[cfg(feature = "gzip")]
fn maybe_gunzip(bytes: Vec<u8>, path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    if !bytes.starts_with(&[0x1f, 0x8b]) {
        return Ok(bytes);
    }
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(bytes.as_slice())
        .read_to_end(&mut out)
        .map_err(|source| Error::Io { path: path.to_owned(), source })?;
    Ok(out)
}

#[cfg(not(feature = "gzip"))]
fn maybe_gunzip(bytes: Vec<u8>, path: &Path) -> Result<Vec<u8>> {
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return Err(Error::Invalid {
            node: "file".into(),
            line: 0,
            message: format!("{} is gzipped; build with the `gzip` feature", path.display()),
        });
    }
    Ok(bytes)
}

/// Parse X3D XML text.
///
/// ```
/// let scene = x3d_scene::parse_str(r#"
///   <X3D><Scene>
///     <Shape><Box size='1 2 3'/></Shape>
///   </Scene></X3D>"#).unwrap();
/// assert_eq!(scene.draws.len(), 1);
/// assert_eq!(scene.report.stats.triangles, 12);
/// ```
pub fn parse_str(xml: &str) -> Result<Scene> {
    parse_str_with(xml, &LoadOptions::default())
}

/// Parse text that came from no file: `Inline` and texture urls can't be
/// resolved (there is no folder to look in), and are reported as such.
pub fn parse_str_with(xml: &str, opts: &LoadOptions) -> Result<Scene> {
    parse_inner(xml, opts, None, &mut InlineState::default())
}

fn parse_inner(xml: &str, opts: &LoadOptions, dir: Option<&Path>, state: &mut InlineState) -> Result<Scene> {
    let head = xml.trim_start();
    if head.starts_with("#X3D") || head.starts_with("#VRML") {
        return Err(Error::Invalid {
            node: "file".into(),
            line: 1,
            message: "this is the ClassicVRML encoding (.x3dv/.wrl); only the XML encoding is supported. \
                      Convert it first (e.g. with castle-model-converter or X_ITE)."
                .into(),
        });
    }
    let doc = Document::parse_with_options(xml, roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() })?;
    let scene_el = doc.descendants().find(|n| n.has_tag_name("Scene")).ok_or(Error::NoScene)?;

    let mut loader = Loader::new(&doc, opts, dir, state);
    loader.children(scene_el, Mat4::IDENTITY, 0)?;
    loader.wire_routes(&doc);
    Ok(loader.finish())
}

// ---------------------------------------------------------------------------
// Small helpers on roxmltree nodes
// ---------------------------------------------------------------------------

/// 🦀 An extension trait: our methods on someone else's type.
trait NodeExt {
    fn name(&self) -> &str;
    fn line(&self) -> u32;
    /// `Ok(None)` if the attribute is absent, `Err` if it doesn't parse.
    fn field<T: FieldValue>(&self, field: &str) -> Result<Option<T>>;
    fn field_or<T: FieldValue>(&self, field: &str, default: T) -> Result<T> {
        Ok(self.field(field)?.unwrap_or(default))
    }
    fn invalid(&self, message: impl Into<String>) -> Error;
    fn label(&self) -> String;
}

impl NodeExt for Node<'_, '_> {
    fn name(&self) -> &str {
        self.tag_name().name()
    }
    fn line(&self) -> u32 {
        self.document().text_pos_at(self.range().start).row
    }
    fn field<T: FieldValue>(&self, field: &str) -> Result<Option<T>> {
        // 🦀 `Option::map` + `transpose` turns Option<Result<T>> into Result<Option<T>>.
        self.attribute(field)
            .map(|text| {
                T::parse_field(text).map_err(|source| Error::Field {
                    node: self.name().to_owned(),
                    field: field.to_owned(),
                    line: self.line(),
                    source,
                })
            })
            .transpose()
    }
    fn invalid(&self, message: impl Into<String>) -> Error {
        Error::Invalid { node: self.name().to_owned(), line: self.line(), message: message.into() }
    }
    fn label(&self) -> String {
        match self.attribute("DEF") {
            Some(def) => format!("{} \"{def}\"", self.name()),
            None => format!("{} (line {})", self.name(), self.line()),
        }
    }
}

/// Child *elements* only (skips text, comments, and metadata nodes).
fn elements<'a, 'i>(n: Node<'a, 'i>) -> impl Iterator<Item = Node<'a, 'i>> {
    n.children().filter(|c| c.is_element() && !c.tag_name().name().starts_with("Metadata"))
}

/// Grouping nodes we treat as plain groups.
const GROUPS: &[&str] =
    &["Group", "StaticGroup", "Collision", "Anchor", "CADAssembly", "CADLayer", "CADPart", "CADFace"];

/// Nodes that only matter for animation or interaction.
fn is_behaviour(name: &str) -> bool {
    const SUFFIXES: &[&str] = &["Sensor", "Interpolator", "Sequencer", "Trigger", "Filter", "Chaser", "Damper"];
    matches!(name, "Script" | "ROUTE" | "TimeSensor" | "Sound" | "AudioClip")
        || SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// Texture nodes (but not TextureProperties / TextureTransform / TextureCoordinate).
fn is_texture(name: &str) -> bool {
    name.ends_with("Texture") || name.ends_with("Texture3D") || name == "ImageTextureAtlas"
}

/// Nodes that carry no picture at all.
fn is_silent(name: &str) -> bool {
    matches!(
        name,
        "WorldInfo"
            | "ProtoDeclare"
            | "ExternProtoDeclare"
            | "IMPORT"
            | "EXPORT"
            | "IS"
            | "connect"
            | "field"
            | "fieldValue"
    )
}

// ---------------------------------------------------------------------------
// The loader
// ---------------------------------------------------------------------------

/// 🦀 Two lifetimes: `'a` for borrows of the document, `'i` for the input
/// text the document itself borrows. `'i: 'a` ("'i outlives 'a") is implied.
/// The `&'a mut InlineState` is *lent* to us by the caller; an Inline lends
/// it on again (a "reborrow") to the nested load, then gets it back.
struct Loader<'a, 'i> {
    defs: HashMap<&'a str, Node<'a, 'i>>,
    opts: &'a LoadOptions,
    /// The folder the scene came from: Inline and texture urls are relative to it.
    dir: Option<&'a Path>,
    inline: &'a mut InlineState,
    scene: Scene,
    /// Geometry node → its mesh, so `USE`d geometry is built once.
    /// `None` = unsupported geometry (remembered so we report it once).
    meshes: HashMap<NodeId, Option<Arc<Mesh>>>,
    seen_navigation: bool,
    seen_background: bool,
    /// The LOD levels we are inside right now.
    lod: Vec<LodTag>,
    /// The light scopes we are inside right now.
    scopes: Vec<u32>,

    // --- for animation ---
    /// The transform node we are inside (`Scene::nodes`); the root is 0.
    node_stack: Vec<u32>,
    /// Animated Switches we are inside.
    switch_guards: Vec<SwitchGuard>,
    /// DEF names some ROUTE writes to: only those Switches stay dynamic.
    route_targets: HashSet<&'a str>,
    /// (Appearance, lines-or-points) → index into `Scene::materials`.
    material_ids: HashMap<(Option<NodeId>, bool), u32>,
    /// Which scene parts each X3D node turned into, for ROUTE targets.
    users: HashMap<NodeId, Vec<u32>>,
    texture_transforms: HashMap<NodeId, (Vec<u32>, TextureTransform2D)>,
    /// X3D node → its node in `Scene::animation`.
    anim_ids: HashMap<NodeId, u32>,
    /// ComposedShader node → index into `Scene::shaders`.
    shader_ids: HashMap<NodeId, u32>,
}

impl<'a, 'i> Loader<'a, 'i> {
    fn new(doc: &'a Document<'i>, opts: &'a LoadOptions, dir: Option<&'a Path>, inline: &'a mut InlineState) -> Self {
        // Index every DEF in the document up front.
        let defs = doc.descendants().filter_map(|n| n.attribute("DEF").map(|d| (d, n))).collect();
        let mut scene = Scene { headlight: true, ..Default::default() };
        scene.nodes.push(TransformNode::root());
        let route_targets =
            doc.descendants().filter(|n| n.has_tag_name("ROUTE")).filter_map(|n| n.attribute("toNode")).collect();
        Self {
            defs,
            opts,
            dir,
            inline,
            scene,
            meshes: HashMap::new(),
            seen_navigation: false,
            seen_background: false,
            lod: Vec::new(),
            scopes: Vec::new(),
            node_stack: vec![0],
            switch_guards: Vec::new(),
            route_targets,
            material_ids: HashMap::new(),
            users: HashMap::new(),
            texture_transforms: HashMap::new(),
            anim_ids: HashMap::new(),
            shader_ids: HashMap::new(),
        }
    }

    /// The transform node we are inside.
    fn node(&self) -> u32 {
        *self.node_stack.last().expect("the root node is never popped")
    }

    /// Remember that X3D node `n` became scene part `id` (for ROUTEs).
    fn used_as(&mut self, n: Node<'a, 'i>, id: u32) {
        self.users.entry(n.id()).or_default().push(id);
    }

    /// Follow `USE='name'` to the node that `DEF`ined it.
    fn resolve(&self, n: Node<'a, 'i>) -> Result<Node<'a, 'i>> {
        match n.attribute("USE") {
            None => Ok(n),
            Some(name) => self.defs.get(name).copied().ok_or_else(|| Error::MissingDef(name.to_owned())),
        }
    }

    /// The first child element (USE resolved) whose name is in `names`.
    fn child(&self, n: Node<'a, 'i>, names: &[&str]) -> Result<Option<Node<'a, 'i>>> {
        for c in elements(n) {
            if names.contains(&c.name()) {
                return self.resolve(c).map(Some);
            }
        }
        Ok(None)
    }

    fn children(&mut self, n: Node<'a, 'i>, world: Mat4, depth: usize) -> Result<()> {
        // A group holding a scoped (global='false') light opens a scope:
        // that light reaches only draws inside this group.
        let scoped = elements(n).filter_map(|c| self.resolve(c).ok()).any(|c| {
            let default_global = !c.has_tag_name("DirectionalLight");
            matches!(c.name(), "DirectionalLight" | "PointLight" | "SpotLight")
                && !c.field_or("global", default_global).unwrap_or(default_global)
        });
        if scoped {
            self.scopes.push(self.scene.light_scopes);
            self.scene.light_scopes += 1;
        }
        let result = elements(n).try_for_each(|c| self.visit(c, world, depth + 1));
        if scoped {
            self.scopes.pop();
        }
        result
    }

    fn visit(&mut self, n: Node<'a, 'i>, world: Mat4, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(Error::TooDeep(MAX_DEPTH));
        }
        let n = self.resolve(n)?;
        // X3D 4: grouping nodes and Shape have `visible`.
        if !n.field_or("visible", true)? {
            return Ok(());
        }
        let name = n.name();
        match name {
            "Transform" => {
                // Every Transform *instance* becomes a node in the transform
                // tree, so a ROUTE can move it (and everything under it) later.
                let trs = trs_of(n)?;
                let world = world * trs.matrix();
                let id = self.scene.nodes.len() as u32;
                self.scene.nodes.push(TransformNode { parent: Some(self.node()), local: Local::Trs(trs), world });
                self.used_as(n, id);
                self.node_stack.push(id);
                let result = self.children(n, world, depth);
                self.node_stack.pop();
                result
            }
            _ if GROUPS.contains(&name) => self.children(n, world, depth),
            "Billboard" => {
                self.scene.report.approximated("Billboard (drawn as a Group, not turned to face the camera)");
                self.children(n, world, depth)
            }
            "LOD" => {
                // Keep every level; the renderer picks one per frame from the
                // camera's distance to `center`.
                let group = self.scene.lod_groups;
                self.scene.lod_groups += 1;
                let center = world.transform_point3(n.field_or("center", Vec3::ZERO)?);
                let range: Arc<[f32]> = n.field_or::<Vec<f32>>("range", Vec::new())?.into();
                for (level, child) in elements(n).enumerate() {
                    let tag = LodTag { group, level: level as u32, center, range: Arc::clone(&range) };
                    self.lod.push(tag);
                    let result = self.visit(child, world, depth + 1);
                    self.lod.pop();
                    result?;
                }
                Ok(())
            }
            "Switch" => {
                // whichChoice = -1 shows nothing. (X3D 4 calls the field `children`, X3D 3 `choice`;
                // in XML both are just child elements.)
                let choice: i32 = n.field_or("whichChoice", -1)?;
                if n.attribute("DEF").is_some_and(|d| self.route_targets.contains(d)) {
                    // A ROUTE can change whichChoice: keep every child, each
                    // guarded by "this Switch is on child i".
                    let id = self.scene.switches.len() as u32;
                    self.scene.switches.push(choice);
                    self.used_as(n, id);
                    for (i, child) in elements(n).enumerate() {
                        self.switch_guards.push(SwitchGuard { switch: id, child: i as u32 });
                        let result = self.visit(child, world, depth + 1);
                        self.switch_guards.pop();
                        result?;
                    }
                    return Ok(());
                }
                match usize::try_from(choice).ok().and_then(|i| elements(n).nth(i)) {
                    Some(chosen) => self.visit(chosen, world, depth + 1),
                    None => Ok(()),
                }
            }
            "Shape" => match self.shape(n, world) {
                // 🦀 A match guard (`if …`) decides per error whether to recover.
                Err(e) if !self.opts.strict => {
                    self.scene.report.warn(format!("skipped a Shape: {e}"));
                    Ok(())
                }
                other => other,
            },
            "DirectionalLight" | "PointLight" | "SpotLight" => self.light(n, world),
            "EnvironmentLight" => self.environment_light(n),
            "Viewpoint" | "OrthoViewpoint" => self.viewpoint(n, world),
            "NavigationInfo" => {
                if !std::mem::replace(&mut self.seen_navigation, true) {
                    self.scene.headlight = n.field_or("headlight", true)?;
                }
                Ok(())
            }
            "Background" => self.background(n),
            "Inline" => self.inline(n, world),
            // TimeSensors, interpolators… are wired up from the ROUTEs at the end.
            _ if is_behaviour(name) => Ok(()),
            _ if is_silent(name) => Ok(()),
            other => {
                self.scene.report.unsupported(other);
                Ok(())
            }
        }
    }

    // --- Shape -------------------------------------------------------------

    fn shape(&mut self, n: Node<'a, 'i>, world: Mat4) -> Result<()> {
        let mut appearance = None;
        let mut geometry = None;
        for c in elements(n) {
            let c = self.resolve(c)?;
            if c.has_tag_name("Appearance") {
                appearance = Some(c);
            } else if geometry.is_none() {
                geometry = Some(c);
            }
        }
        // A Shape with no geometry is legal and draws nothing.
        let Some(g) = geometry else { return Ok(()) };
        let Some(mesh) = self.mesh_for(g)? else {
            return Ok(());
        };
        let material_id = self.material_id(appearance, &mesh)?;
        self.scene.draws.push(Draw {
            lod: self.lod.as_slice().into(),
            scopes: self.scopes.as_slice().into(),
            node: self.node(),
            material: self.scene.materials[material_id as usize].clone(),
            material_id,
            switches: self.switch_guards.as_slice().into(),
            mesh,
            transform: world,
        });
        Ok(())
    }

    fn mesh_for(&mut self, g: Node<'a, 'i>) -> Result<Option<Arc<Mesh>>> {
        if let Some(cached) = self.meshes.get(&g.id()) {
            return Ok(cached.clone()); // cloning an Arc = bumping a counter
        }
        let built = self.build_geometry(g)?.map(Arc::new);
        self.meshes.insert(g.id(), built.clone());
        Ok(built)
    }

    fn build_geometry(&mut self, g: Node<'a, 'i>) -> Result<Option<Mesh>> {
        let segments = self.opts.segments;
        let parts = |side: bool, top: bool, bottom: bool| Parts { side, top, bottom };
        let mut mesh = match g.name() {
            "Box" => geometry::cuboid(g.field_or("size", Vec3::splat(2.0))?),
            "Sphere" => geometry::sphere(g.field_or("radius", 1.0)?, segments),
            "Cylinder" => geometry::cylinder(
                g.field_or("radius", 1.0)?,
                g.field_or("height", 2.0)?,
                parts(g.field_or("side", true)?, g.field_or("top", true)?, g.field_or("bottom", true)?),
                segments,
            ),
            "Cone" => geometry::cone(
                g.field_or("bottomRadius", 1.0)?,
                g.field_or("height", 2.0)?,
                parts(g.field_or("side", true)?, false, g.field_or("bottom", true)?),
                segments,
            ),
            "IndexedFaceSet" => self.indexed_face_set(g)?.tessellate(),
            "TriangleSet"
            | "IndexedTriangleSet"
            | "TriangleFanSet"
            | "IndexedTriangleFanSet"
            | "TriangleStripSet"
            | "IndexedTriangleStripSet"
            | "QuadSet"
            | "IndexedQuadSet" => self.triangle_family(g)?.tessellate(),
            "ElevationGrid" => self.elevation_grid(g)?.tessellate(),
            "IndexedLineSet" | "LineSet" => self.lines(g)?,
            "PointSet" => {
                let points = self.points(g)?;
                let colors = self.colors(g)?;
                geometry::point_set(&points, colors.as_deref())
            }
            other => {
                self.scene.report.unsupported(other);
                return Ok(None);
            }
        };
        if mesh.topology == Topology::Triangles {
            mesh.solid = g.field_or("solid", true)?;
        }
        mesh.rgba_colors = self.child(g, &["ColorRGBA"])?.is_some();
        mesh.label = g.label();
        Ok(Some(mesh))
    }

    fn points(&self, g: Node<'a, 'i>) -> Result<Vec<Vec3>> {
        match self.child(g, &["Coordinate", "CoordinateDouble"])? {
            Some(c) => c.field_or("point", Vec::new()),
            None => Ok(Vec::new()),
        }
    }

    /// Color (RGB) or ColorRGBA (RGBA) → RGBA.
    fn colors(&self, g: Node<'a, 'i>) -> Result<Option<Vec<Vec4>>> {
        Ok(match self.child(g, &["Color", "ColorRGBA"])? {
            Some(c) if c.has_tag_name("Color") => {
                let rgb: Vec<Vec3> = c.field_or("color", Vec::new())?;
                Some(rgb.into_iter().map(|c| c.extend(1.0)).collect())
            }
            Some(c) => Some(c.field_or("color", Vec::new())?),
            None => None,
        })
    }

    /// Texture coordinates for each point: explicit ones, or positions for
    /// `TextureCoordinateGenerator` (the texture matrix then projects them).
    fn texcoords(&mut self, g: Node<'a, 'i>, points: &[Vec3]) -> Result<Option<Vec<Vec3>>> {
        const KINDS: &[&str] =
            &["TextureCoordinate", "TextureCoordinate3D", "TextureCoordinateGenerator", "MultiTextureCoordinate"];
        let Some(mut tc) = self.child(g, KINDS)? else {
            return Ok(None);
        };
        if tc.has_tag_name("MultiTextureCoordinate") {
            if elements(tc).count() > 1 {
                self.scene.report.approximated("MultiTextureCoordinate (first set only)");
            }
            match elements(tc).next() {
                Some(first) => tc = self.resolve(first)?,
                None => return Ok(None),
            }
        }
        Ok(Some(match tc.name() {
            "TextureCoordinate" => {
                let st: Vec<Vec2> = tc.field_or("point", Vec::new())?;
                st.into_iter().map(|p| p.extend(0.0)).collect()
            }
            "TextureCoordinate3D" => tc.field_or("point", Vec::new())?,
            "TextureCoordinateGenerator" => {
                let mode = tc.attribute("mode").unwrap_or("SPHERE");
                if mode != "COORD" {
                    self.scene
                        .report
                        .approximated(&format!("TextureCoordinateGenerator mode='{mode}' (used as COORD)"));
                }
                points.to_vec()
            }
            _ => return Ok(None),
        }))
    }

    fn normals(&self, g: Node<'a, 'i>) -> Result<Vec<Vec3>> {
        match self.child(g, &["Normal"])? {
            Some(n) => n.field_or("vector", Vec::new()),
            None => Ok(Vec::new()),
        }
    }

    /// Attach colours/normals and check indices; shared by every polygon geometry.
    fn finish_polygons(
        &mut self,
        g: Node<'a, 'i>,
        points: Vec<Vec3>,
        faces: Vec<Vec<u32>>,
        index: &Indices,
        crease_angle: f32,
    ) -> Result<Polygons> {
        let at = |e: IndexError| g.invalid(e.to_string());
        // 🦀 Destructuring a struct into local names in one `let`.
        let Indices { color: color_index, normal: normal_index, tex: tex_index } = *index;
        let texcoords = match self.texcoords(g, &points)? {
            Some(values) => geometry::bind(&values, true, tex_index, &faces, "texCoord").map_err(at)?,
            None => geometry::Binding::None,
        };
        let colors = self.colors(g)?.unwrap_or_default();
        let normals = self.normals(g)?;
        let colors =
            geometry::bind(&colors, g.field_or("colorPerVertex", true)?, color_index, &faces, "color").map_err(at)?;
        let normals = geometry::bind(&normals, g.field_or("normalPerVertex", true)?, normal_index, &faces, "normal")
            .map_err(at)?;
        let polys = Polygons { points, faces, colors, normals, texcoords, ccw: g.field_or("ccw", true)?, crease_angle };
        polys.validate().map_err(at)?;
        Ok(polys)
    }

    fn indexed_face_set(&mut self, g: Node<'a, 'i>) -> Result<Polygons> {
        let coord_index: Vec<i32> = g.field_or("coordIndex", Vec::new())?;
        let faces = to_faces(&coord_index);
        if !g.field_or("convex", true)? && faces.iter().any(|f| f.len() > 3) {
            self.scene
                .report
                .approximated("IndexedFaceSet convex='false' (fan-triangulated; see the ear-clipping quest)");
        }
        self.finish_polygons(
            g,
            self.points(g)?,
            faces,
            &Indices {
                color: &g.field_or::<Vec<i32>>("colorIndex", Vec::new())?,
                normal: &g.field_or::<Vec<i32>>("normalIndex", Vec::new())?,
                tex: &g.field_or::<Vec<i32>>("texCoordIndex", Vec::new())?,
            },
            g.field_or("creaseAngle", 0.0)?,
        )
    }

    /// All the TriangleSet/Fan/Strip/Quad variants become triangle (or quad) faces.
    fn triangle_family(&mut self, g: Node<'a, 'i>) -> Result<Polygons> {
        let points = self.points(g)?;
        let name = g.name();
        let indexed = name.starts_with("Indexed");
        let index: Vec<u32> = if indexed {
            let raw: Vec<i32> = g.field_or("index", Vec::new())?;
            // Strip/fan sets keep their -1 separators; handled below.
            raw.iter().map(|&i| if i < 0 { u32::MAX } else { i as u32 }).collect()
        } else {
            (0..points.len() as u32).collect()
        };
        let runs = |counts: &[i32]| -> Vec<Vec<u32>> {
            let mut out = Vec::new();
            let mut start = 0usize;
            for &c in counts {
                let c = c.max(0) as usize;
                out.push(index.get(start..start + c).unwrap_or(&[]).to_vec());
                start += c;
            }
            out
        };
        let separated = || -> Vec<Vec<u32>> {
            index.split(|&i| i == u32::MAX).filter(|r| !r.is_empty()).map(<[u32]>::to_vec).collect()
        };
        let faces: Vec<Vec<u32>> = match name {
            "TriangleSet" | "IndexedTriangleSet" => index.chunks_exact(3).map(<[u32]>::to_vec).collect(),
            "QuadSet" | "IndexedQuadSet" => index.chunks_exact(4).map(<[u32]>::to_vec).collect(),
            "TriangleFanSet" | "IndexedTriangleFanSet" => {
                let fans = if indexed { separated() } else { runs(&g.field_or("fanCount", Vec::new())?) };
                fans.iter()
                    .flat_map(|f| (1..f.len().saturating_sub(1)).map(move |k| vec![f[0], f[k], f[k + 1]]))
                    .collect()
            }
            _ => {
                let strips = if indexed { separated() } else { runs(&g.field_or("stripCount", Vec::new())?) };
                strips
                    .iter()
                    .flat_map(|s| {
                        (0..s.len().saturating_sub(2)).map(move |k| {
                            // Every other strip triangle is wound the other way.
                            if k % 2 == 0 { vec![s[k], s[k + 1], s[k + 2]] } else { vec![s[k + 1], s[k], s[k + 2]] }
                        })
                    })
                    .collect()
            }
        };
        self.finish_polygons(g, points, faces, &Indices::default(), 0.0)
    }

    fn elevation_grid(&mut self, g: Node<'a, 'i>) -> Result<Polygons> {
        let nx = g.field_or::<i32>("xDimension", 0)?.max(0) as usize;
        let nz = g.field_or::<i32>("zDimension", 0)?.max(0) as usize;
        let (dx, dz): (f32, f32) = (g.field_or("xSpacing", 1.0)?, g.field_or("zSpacing", 1.0)?);
        let mut heights: Vec<f32> = g.field_or("height", Vec::new())?;
        if heights.len() < nx * nz {
            self.scene.report.warn(format!(
                "{}: {} heights for a {nx}×{nz} grid; the missing ones are 0",
                g.label(),
                heights.len()
            ));
            heights.resize(nx * nz, 0.0);
        }
        let points = (0..nz)
            .flat_map(|j| (0..nx).map(move |i| (i, j)))
            .map(|(i, j)| Vec3::new(i as f32 * dx, heights[i + j * nx], j as f32 * dz))
            .collect();
        let id = |i: usize, j: usize| (i + j * nx) as u32;
        let faces = (0..nz.saturating_sub(1))
            .flat_map(|j| (0..nx.saturating_sub(1)).map(move |i| (i, j)))
            .map(|(i, j)| vec![id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j)])
            .collect();
        self.finish_polygons(g, points, faces, &Indices::default(), g.field_or("creaseAngle", 0.0)?)
    }

    fn lines(&mut self, g: Node<'a, 'i>) -> Result<Mesh> {
        let points = self.points(g)?;
        let lines: Vec<Vec<u32>> = if g.has_tag_name("IndexedLineSet") {
            to_faces(&g.field_or("coordIndex", Vec::new())?)
        } else {
            let counts: Vec<i32> = g.field_or("vertexCount", Vec::new())?;
            let mut start = 0u32;
            counts
                .iter()
                .map(|&c| {
                    let run = (start..start + c.max(0) as u32).collect();
                    start += c.max(0) as u32;
                    run
                })
                .collect()
        };
        let colors = self.colors(g)?.unwrap_or_default();
        let color_index: Vec<i32> = g.field_or("colorIndex", Vec::new())?;
        let colors = geometry::bind(&colors, g.field_or("colorPerVertex", true)?, &color_index, &lines, "color")
            .map_err(|e| g.invalid(e.to_string()))?;
        if let Some(&bad) = lines.iter().flatten().find(|&&c| c as usize >= points.len()) {
            return Err(g.invalid(format!("coordinate index {bad} is out of range ({} points)", points.len())));
        }
        Ok(geometry::polylines(&points, &lines, &colors))
    }

    // --- Appearance --------------------------------------------------------

    /// The material-table entry for an Appearance (shared by every Shape
    /// that USEs it, so one event recolours them all).
    fn material_id(&mut self, appearance: Option<Node<'a, 'i>>, mesh: &Mesh) -> Result<u32> {
        let key = (appearance.map(|a| a.id()), mesh.topology != Topology::Triangles);
        if let Some(&id) = self.material_ids.get(&key) {
            return Ok(id);
        }
        let material = self.material(appearance, mesh)?;
        let id = self.scene.materials.len() as u32;
        self.scene.materials.push(material);
        self.material_ids.insert(key, id);
        if let Some(app) = appearance {
            const MATERIALS: &[&str] = &["Material", "UnlitMaterial", "PhysicalMaterial", "TwoSidedMaterial"];
            if let Some(m) = self.child(app, MATERIALS)? {
                self.used_as(m, id);
            }
            if let Some(mut t) = self.child(app, &["TextureTransform", "MultiTextureTransform"])? {
                if t.has_tag_name("MultiTextureTransform") {
                    if let Some(first) = elements(t).next() {
                        t = self.resolve(first)?;
                    }
                }
                if t.has_tag_name("TextureTransform") {
                    let fields = TextureTransform2D {
                        center: t.field_or("center", Vec2::ZERO)?,
                        rotation: t.field_or("rotation", 0.0)?,
                        scale: t.field_or("scale", Vec2::ONE)?,
                        translation: t.field_or("translation", Vec2::ZERO)?,
                    };
                    self.texture_transforms.entry(t.id()).or_insert_with(|| (Vec::new(), fields)).0.push(id);
                }
            }
        }
        Ok(id)
    }

    fn material(&mut self, appearance: Option<Node<'a, 'i>>, mesh: &Mesh) -> Result<Material> {
        let lines_or_points = mesh.topology != Topology::Triangles;
        let Some(app) = appearance else {
            return Ok(Material::UNLIT_WHITE);
        };

        for c in elements(app) {
            match c.name() {
                "ProgramShader" | "PackagedShader" => {
                    self.scene.report.unsupported(&format!("{} (custom shaders)", c.name()));
                }
                _ => {}
            }
        }

        let m = self.child(app, &["Material", "UnlitMaterial", "PhysicalMaterial", "TwoSidedMaterial"])?;
        let mut out = match m {
            None => Material::UNLIT_WHITE,
            Some(m) => self.material_node(m)?,
        };
        if let Some(m) = m {
            out.textures = self.material_textures(m)?;
        }

        // Appearance.texture: X3D's original texture slot. It *replaces* a
        // classic Material's diffuse colour, and tints an unlit one.
        for c in elements(app) {
            let field = c.attribute("containerField").unwrap_or("texture");
            if is_texture(c.name()) && field == "texture" && out.textures.base.is_none() {
                let c = self.resolve(c)?;
                out.textures.base = self.texture(c)?;
                if out.textures.base.is_some() && m.is_some_and(|m| m.has_tag_name("Material")) {
                    out.diffuse = Vec3::ONE;
                }
            }
        }
        out.texture_matrix = self.texture_matrix(app)?;
        // X3D lists shaders in order of preference: take the first GLSL one.
        let shaders: Vec<_> = elements(app).filter(|c| c.has_tag_name("ComposedShader")).collect();
        for c in shaders {
            let c = self.resolve(c)?;
            if c.attribute("language").unwrap_or("GLSL") == "GLSL" {
                out.shader = self.composed_shader(c)?;
                break;
            }
        }
        // Line width / point size (X3D: a multiple of the 1 px nominal size).
        if let Some(p) = self.child(app, &["LineProperties", "PointProperties"])? {
            let (field, applied) = match p.name() {
                "LineProperties" => ("linewidthScaleFactor", p.field_or("applied", true)?),
                _ => ("pointSizeScaleFactor", true),
            };
            if applied {
                out.line_width = p.field_or(field, 1.0f32)?.max(1.0);
            }
        }

        if lines_or_points {
            // X3D: lines and points are unlit, drawn in emissiveColor.
            out.unlit = true;
        }
        Ok(out)
    }

    fn material_node(&mut self, m: Node<'a, 'i>) -> Result<Material> {
        let mut out = Material::default();
        match m.name() {
            "Material" | "TwoSidedMaterial" => {
                if m.has_tag_name("TwoSidedMaterial") && m.field_or("separateBackColor", false)? {
                    self.scene.report.approximated("TwoSidedMaterial back colours (front used on both sides)");
                }
                out.diffuse = m.field_or("diffuseColor", out.diffuse)?;
                out.emissive = m.field_or("emissiveColor", out.emissive)?;
                out.specular = m.field_or("specularColor", out.specular)?;
                out.shininess = m.field_or("shininess", out.shininess)?;
                out.ambient_intensity = m.field_or("ambientIntensity", out.ambient_intensity)?;
                out.transparency = m.field_or("transparency", 0.0)?;
            }
            "UnlitMaterial" => {
                out.emissive = m.field_or("emissiveColor", Vec3::ONE)?;
                out.diffuse = out.emissive;
                out.transparency = m.field_or("transparency", 0.0)?;
                out.unlit = true;
            }
            _ => {
                // PhysicalMaterial: glTF-style metallic-roughness. X3D's defaults
                // are metallic = roughness = 1 (a rough metal).
                out.diffuse = m.field_or("baseColor", Vec3::ONE)?;
                out.emissive = m.field_or("emissiveColor", Vec3::ZERO)?;
                out.transparency = m.field_or("transparency", 0.0)?;
                out.pbr = Some(Pbr {
                    metallic: m.field_or("metallic", 1.0)?,
                    roughness: m.field_or("roughness", 1.0)?,
                    occlusion_strength: m.field_or("occlusionStrength", 1.0)?,
                    normal_scale: m.field_or("normalScale", 1.0)?,
                });
            }
        }
        Ok(out)
    }

    /// X3D 4 puts textures *inside* the material, each named by its
    /// `containerField`. Note: we read `containerField` from the element as
    /// written — a `USE` element carries its own.
    fn material_textures(&mut self, m: Node<'a, 'i>) -> Result<Textures> {
        let mut t = Textures::default();
        let unlit = m.has_tag_name("UnlitMaterial");
        for c in elements(m) {
            if !is_texture(c.name()) {
                continue;
            }
            let field = c.attribute("containerField").unwrap_or_default();
            let slot = match field {
                "baseTexture" | "diffuseTexture" => &mut t.base,
                "emissiveTexture" if unlit => &mut t.base,
                "emissiveTexture" => &mut t.emissive,
                "normalTexture" => &mut t.normal,
                "metallicRoughnessTexture" => &mut t.metallic_roughness,
                "occlusionTexture" => &mut t.occlusion,
                other => {
                    self.scene.report.unsupported(&format!("{} {other}", m.name()));
                    continue;
                }
            };
            let resolved = self.resolve(c)?;
            *slot = self.texture(resolved)?;
        }
        Ok(t)
    }

    /// A ComposedShader → an entry in `Scene::shaders` (once per node, so a
    /// USE'd shader shares its fields).
    fn composed_shader(&mut self, n: Node<'a, 'i>) -> Result<Option<u32>> {
        if let Some(&id) = self.shader_ids.get(&n.id()) {
            return Ok(Some(id));
        }
        let mut vertex = None;
        let mut fragment = None;
        let parts: Vec<_> = elements(n).filter(|p| p.has_tag_name("ShaderPart")).collect();
        for part in parts {
            let part = self.resolve(part)?;
            let source = self.shader_part_source(part)?;
            match part.attribute("type").unwrap_or("VERTEX") {
                "VERTEX" => vertex = source,
                "FRAGMENT" => fragment = source,
                other => self.scene.report.unsupported(&format!("ShaderPart type='{other}'")),
            }
        }
        let (Some(vertex), Some(fragment)) = (vertex, fragment) else {
            self.scene.report.warn(format!("{}: needs a VERTEX and a FRAGMENT part", n.label()));
            return Ok(None);
        };
        // `<field name='time' type='SFTime' value='0'/>`: uniforms that events can set.
        let mut fields = Vec::new();
        for f in elements(n).filter(|f| f.has_tag_name("field")) {
            let Some(name) = f.attribute("name") else { continue };
            let value = f.attribute("value").unwrap_or("");
            let parsed = match f.attribute("type").unwrap_or("SFFloat") {
                "SFFloat" => f32::parse_field(value).ok().map(Value::Float),
                "SFTime" | "SFDouble" => value.trim().parse().ok().map(Value::Time),
                "SFInt32" => i32::parse_field(value).ok().map(Value::Int),
                "SFBool" => bool::parse_field(value).ok().map(Value::Bool),
                "SFVec2f" => Vec2::parse_field(value).ok().map(Value::Vec2),
                "SFVec3f" | "SFColor" => Vec3::parse_field(value).ok().map(Value::Vec3),
                "SFVec4f" | "SFColorRGBA" => Vec4::parse_field(value).ok().map(Value::Vec4),
                "SFRotation" => Quat::parse_field(value).ok().map(Value::Rotation),
                other => {
                    self.scene.report.unsupported(&format!("ComposedShader field of type {other}"));
                    None
                }
            };
            // A field without a value starts at zero, like an unset uniform.
            fields.push((name.to_owned(), self.scene.shader_fields.len() as u32));
            self.scene.shader_fields.push(parsed.unwrap_or(Value::Float(0.0)));
        }
        let id = self.scene.shaders.len() as u32;
        self.scene.shaders.push(ShaderSource { label: n.label(), vertex, fragment, fields });
        self.shader_ids.insert(n.id(), id);
        Ok(Some(id))
    }

    /// A ShaderPart's GLSL: inline text (CDATA), a `data:` url, or a file.
    fn shader_part_source(&mut self, part: Node<'a, 'i>) -> Result<Option<String>> {
        let strip_data = |s: &str| -> String {
            match s.trim_start().strip_prefix("data:") {
                Some(rest) => rest.split_once(',').map_or(rest, |(_, src)| src).to_owned(),
                None => s.to_owned(),
            }
        };
        // Inline text: X3D XML usually puts it in CDATA (roxmltree: text nodes).
        let text: String = part.children().filter_map(|c| c.text()).collect();
        if !text.trim().is_empty() {
            return Ok(Some(strip_data(&text)));
        }
        let urls: Vec<String> = part.field_or("url", Vec::new())?;
        if let Some(data) = urls.iter().find(|u| u.starts_with("data:")) {
            return Ok(Some(strip_data(data)));
        }
        if let Some(dir) = self.dir {
            if let Some(path) = urls.iter().map(|u| dir.join(u)).find(|p| p.is_file()) {
                return Ok(std::fs::read_to_string(&path).ok());
            }
        }
        self.scene.report.warn(format!("{}: no shader source among {urls:?}", part.label()));
        Ok(None)
    }

    /// An ImageTexture → a file on disk.
    fn texture(&mut self, el: Node<'a, 'i>) -> Result<Option<TextureRef>> {
        if !el.has_tag_name("ImageTexture") {
            self.scene.report.unsupported(&format!("{} (only ImageTexture so far)", el.name()));
            return Ok(None);
        }
        let urls: Vec<String> = el.field_or("url", Vec::new())?;
        // 🦀 `data:` urls carry the image itself (base64), no folder needed.
        if let Some(bytes) = urls.iter().find_map(|u| data_url_bytes(u)) {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bytes.hash(&mut h);
            return Ok(Some(TextureRef {
                path: PathBuf::from(format!("data:{:016x}", h.finish())),
                repeat_s: el.field_or("repeatS", true)?,
                repeat_t: el.field_or("repeatT", true)?,
                inline: Some(bytes.into()),
            }));
        }
        let Some(dir) = self.dir else {
            self.scene.report.unsupported("ImageTexture (no folder: the scene was parsed from a string)");
            return Ok(None);
        };
        let found = urls
            .iter()
            .filter(|u| !u.contains("://") && !u.starts_with("data:"))
            .map(|u| dir.join(u))
            .find(|p| p.is_file());
        let Some(path) = found else {
            self.scene.report.warn(format!("{}: no texture file among {urls:?}", el.label()));
            return Ok(None);
        };
        Ok(Some(TextureRef {
            path: path.canonicalize().unwrap_or(path),
            repeat_s: el.field_or("repeatS", true)?,
            repeat_t: el.field_or("repeatT", true)?,
            inline: None,
        }))
    }

    /// `ComposedCubeMapTexture` → six faces in GPU order (+X −X +Y −Y +Z −Z).
    /// X3D 4 field names (`frontTexture`…) and X3D 3 ones (`front`…) both work.
    fn cube_map(&mut self, el: Node<'a, 'i>) -> Result<Option<CubeMap>> {
        if !el.has_tag_name("ComposedCubeMapTexture") {
            self.scene.report.unsupported(&format!("{} (only ComposedCubeMapTexture so far)", el.name()));
            return Ok(None);
        }
        // X3D: front = −Z, back = +Z, left = −X, right = +X, top = +Y, bottom = −Y.
        let order = ["right", "left", "top", "bottom", "back", "front"];
        let mut faces = Vec::with_capacity(6);
        for face in order {
            let long = format!("{face}Texture");
            let child = el
                .children()
                .find(|c| c.is_element() && matches!(c.attribute("containerField"), Some(f) if f == face || f == long));
            let Some(child) = child else {
                self.scene.report.warn(format!("{}: no {long}", el.label()));
                return Ok(None);
            };
            let Some(t) = self.texture(child)? else { return Ok(None) };
            faces.push(t);
        }
        let Ok(faces) = <[TextureRef; 6]>::try_from(faces) else { return Ok(None) };
        Ok(Some(CubeMap { faces }))
    }

    /// X3D 4.1 `EnvironmentLight`: the first one that is on lights the scene.
    fn environment_light(&mut self, n: Node<'a, 'i>) -> Result<()> {
        let field = |name: &str| n.children().find(|c| c.is_element() && c.attribute("containerField") == Some(name));
        let diffuse = match field("diffuseTexture") {
            Some(c) => self.cube_map(c)?,
            None => None,
        };
        let specular = match field("specularTexture") {
            Some(c) => self.cube_map(c)?,
            None => None,
        };
        let env = Environment {
            color: n.field_or("color", Vec3::ONE)?,
            intensity: n.field_or("intensity", 1.0)?,
            on: n.field_or("on", true)?,
            rotation: n.field_or("rotation", glam::Quat::IDENTITY)?,
            diffuse,
            specular,
        };
        if n.attribute("diffuseCoefficients").is_some() {
            self.scene.report.approximated("EnvironmentLight diffuseCoefficients (ignored; the cube maps are used)");
        }
        if env.diffuse.is_none() && env.specular.is_none() {
            self.scene.report.unsupported("EnvironmentLight without a ComposedCubeMapTexture");
        } else if self.scene.environment.is_none() && env.on {
            self.scene.environment = Some(env);
        }
        Ok(())
    }

    /// TextureTransform (2D), TextureTransform3D or TextureTransformMatrix3D.
    fn texture_matrix(&mut self, app: Node<'a, 'i>) -> Result<Mat4> {
        const KINDS: &[&str] =
            &["TextureTransform", "TextureTransform3D", "TextureTransformMatrix3D", "MultiTextureTransform"];
        let Some(mut t) = self.child(app, KINDS)? else {
            return Ok(Mat4::IDENTITY);
        };
        if t.has_tag_name("MultiTextureTransform") {
            match elements(t).next() {
                Some(first) => t = self.resolve(first)?,
                None => return Ok(Mat4::IDENTITY),
            }
        }
        // X3D: Tc' = −C × S × R × C × T × Tc (read right to left: translate,
        // move the centre to the origin, rotate, scale, move back).
        let compose = |c: Vec3, s: Vec3, r: Quat, t: Vec3| {
            Mat4::from_translation(-c)
                * Mat4::from_scale(s)
                * Mat4::from_quat(r)
                * Mat4::from_translation(c)
                * Mat4::from_translation(t)
        };
        Ok(match t.name() {
            "TextureTransform" => {
                let c: Vec2 = t.field_or("center", Vec2::ZERO)?;
                let s: Vec2 = t.field_or("scale", Vec2::ONE)?;
                let tr: Vec2 = t.field_or("translation", Vec2::ZERO)?;
                let r = Quat::from_rotation_z(t.field_or("rotation", 0.0)?);
                compose(c.extend(0.0), s.extend(1.0), r, tr.extend(0.0))
            }
            "TextureTransform3D" => compose(
                t.field_or("center", Vec3::ZERO)?,
                t.field_or("scale", Vec3::ONE)?,
                t.field_or("rotation", Quat::IDENTITY)?,
                t.field_or("translation", Vec3::ZERO)?,
            ),
            "TextureTransformMatrix3D" => {
                let m: Vec<f32> = t.field_or("matrix", Mat4::IDENTITY.to_cols_array().to_vec())?;
                match <[f32; 16]>::try_from(m) {
                    // SFMatrix4f is column-major: the translation is in the last four.
                    Ok(cols) => Mat4::from_cols_array(&cols),
                    Err(m) => {
                        return Err(t.invalid(format!("matrix has {} numbers, not 16", m.len())));
                    }
                }
            }
            _ => Mat4::IDENTITY,
        })
    }

    // --- Inline ------------------------------------------------------------

    /// Load another X3D file and place its contents here.
    fn inline(&mut self, n: Node<'a, 'i>, world: Mat4) -> Result<()> {
        if !n.field_or("load", true)? && !self.opts.load_all_inlines {
            self.scene.report.ignored("Inline load='false' (--load-all-inlines loads it)");
            return Ok(());
        }
        let Some(dir) = self.dir else {
            self.scene.report.unsupported("Inline (no folder: the scene was parsed from a string)");
            return Ok(());
        };
        let urls: Vec<String> = n.field_or("url", Vec::new())?;
        let found = urls.iter().filter(|u| !u.contains("://")).map(|u| dir.join(u)).find(|p| p.is_file());
        let Some(path) = found else {
            self.scene.report.warn(format!("{}: no file among {urls:?}", n.label()));
            return Ok(());
        };
        let path = path.canonicalize().unwrap_or(path);
        let name = path.file_name().map_or_else(|| path.display().to_string(), |f| f.to_string_lossy().into_owned());
        if self.inline.stack.contains(&path) {
            self.scene.report.warn(format!("{name}: Inline cycle; not loaded again"));
            return Ok(());
        }
        let child = match self.inline.cache.get(&path) {
            Some(scene) => Arc::clone(scene),
            None => {
                self.inline.stack.push(path.clone());
                // 🦀 `self.inline` is `&mut InlineState`; passing it on lends it to
                // the nested load until that call returns.
                let loaded = load_file(&path, self.opts, self.inline);
                self.inline.stack.pop();
                match loaded {
                    Ok(scene) => {
                        let scene = Arc::new(scene);
                        self.inline.cache.insert(path, Arc::clone(&scene));
                        scene
                    }
                    Err(e) if !self.opts.strict => {
                        self.scene.report.warn(format!("{name}: {e}"));
                        return Ok(());
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        let scopes = self.scopes.clone();
        self.scene.splice(&child, world, self.node(), &scopes, &name);
        Ok(())
    }

    // --- Environment -------------------------------------------------------

    fn light(&mut self, n: Node<'a, 'i>, world: Mat4) -> Result<()> {
        let on = n.field_or("on", true)?;
        let color = n.field_or("color", Vec3::ONE)?;
        let intensity = n.field_or("intensity", 1.0)?;
        let ambient_intensity = n.field_or("ambientIntensity", 0.0)?;
        // Kept in the light's own space (events change these fields);
        // `kind` is the same light placed in the world.
        let local = match n.name() {
            "DirectionalLight" => LightKind::Directional { direction: n.field_or("direction", Vec3::NEG_Z)? },
            "PointLight" => LightKind::Point {
                location: n.field_or("location", Vec3::ZERO)?,
                attenuation: n.field_or("attenuation", Vec3::X)?,
                radius: n.field_or("radius", 100.0)?,
            },
            _ => LightKind::Spot {
                location: n.field_or("location", Vec3::ZERO)?,
                direction: n.field_or("direction", Vec3::NEG_Z)?,
                attenuation: n.field_or("attenuation", Vec3::X)?,
                radius: n.field_or("radius", 100.0)?,
                beam_width: n.field_or("beamWidth", std::f32::consts::FRAC_PI_4)?,
                cut_off_angle: n.field_or("cutOffAngle", std::f32::consts::FRAC_PI_2)?,
            },
        };
        // X3D: DirectionalLights default to lighting only their own group
        // (global='false'); point and spot lights default to global.
        let global = n.field_or("global", !n.has_tag_name("DirectionalLight"))?;
        let scope = if global { None } else { self.scopes.last().copied() };
        let id = self.scene.lights.len() as u32;
        self.scene.lights.push(Light {
            kind: local.transformed(&world),
            local,
            color,
            intensity,
            ambient_intensity,
            scope,
            on,
            shadows: n.field_or("shadows", false)?,
            shadow_intensity: n.field_or("shadowIntensity", 1.0f32)?.clamp(0.0, 1.0),
            node: self.node(),
        });
        self.used_as(n, id);
        Ok(())
    }

    fn viewpoint(&mut self, n: Node<'a, 'i>, world: Mat4) -> Result<()> {
        if n.has_tag_name("OrthoViewpoint") {
            self.scene.report.approximated("OrthoViewpoint (shown in perspective)");
        }
        let d = Viewpoint::default();
        let (_, rotation, _) = world.to_scale_rotation_translation();
        let index = self.scene.viewpoints.len() as u32;
        self.used_as(n, index);
        self.scene.viewpoints.push(Viewpoint {
            description: n.attribute("description").unwrap_or_default().to_owned(),
            position: world.transform_point3(n.field_or("position", d.position)?),
            orientation: rotation * n.field_or("orientation", Quat::IDENTITY)?,
            field_of_view: n.field_or("fieldOfView", d.field_of_view)?,
            center_of_rotation: world.transform_point3(n.field_or("centerOfRotation", Vec3::ZERO)?),
        });
        Ok(())
    }

    fn background(&mut self, n: Node<'a, 'i>) -> Result<()> {
        if std::mem::replace(&mut self.seen_background, true) {
            return Ok(()); // only the first Background is bound
        }
        let b = Background {
            sky_colors: n.field_or("skyColor", vec![Vec3::ZERO])?,
            sky_angles: n.field_or("skyAngle", Vec::new())?,
            ground_colors: n.field_or("groundColor", Vec::new())?,
            ground_angles: n.field_or("groundAngle", Vec::new())?,
            transparency: n.field_or("transparency", 0.0)?,
        };
        if b.sky_colors.len() > 1 || !b.ground_colors.is_empty() {
            self.scene.report.approximated("Background gradient (zenith colour only; see the sky quest)");
        }
        if ["frontUrl", "backUrl", "leftUrl", "rightUrl", "topUrl", "bottomUrl"]
            .iter()
            .any(|u| n.attribute(*u).is_some())
        {
            self.scene.report.unsupported("Background panorama images");
        }
        self.scene.background = b;
        Ok(())
    }

    // --- ROUTEs -------------------------------------------------------------

    /// Turn every ROUTE into an edge of the event graph. Routes whose ends
    /// we can't animate are counted in the report.
    fn wire_routes(&mut self, doc: &'a Document<'i>) {
        let inside_proto = |n: &Node| n.ancestors().any(|a| a.has_tag_name("ProtoDeclare"));
        for r in doc.descendants().filter(|n| n.has_tag_name("ROUTE") && !inside_proto(n)) {
            let (Some(from), Some(from_field), Some(to), Some(to_field)) =
                (r.attribute("fromNode"), r.attribute("fromField"), r.attribute("toNode"), r.attribute("toField"))
            else {
                continue;
            };
            let kind = |def: &str| self.defs.get(def).map_or("?", |n| n.tag_name().name());
            let (from_kind, to_kind) = (kind(from), kind(to));
            match (self.anim_node(from), self.anim_node(to)) {
                (Some(from), Some(to)) => self.scene.animation.routes.push(Route {
                    from,
                    from_field: field_name(from_field).to_owned(),
                    to,
                    to_field: field_name(to_field).to_owned(),
                }),
                (None, _) => self.scene.report.ignored(&format!("ROUTE from {from_kind}")),
                (_, None) => self.scene.report.ignored(&format!("ROUTE to {to_kind}.{}", field_name(to_field))),
            }
        }
    }

    /// The event-graph node for a DEF name, created on first use.
    fn anim_node(&mut self, def: &str) -> Option<u32> {
        let n = *self.defs.get(def)?;
        if let Some(&id) = self.anim_ids.get(&n.id()) {
            return Some(id);
        }
        // Lenient: a field that doesn't parse falls back to its default.
        let key: Vec<f32> = n.field_or("key", Vec::new()).unwrap_or_default();
        let interp = |values| AnimNode::Behaviour(Behaviour::Interpolator { key: key.clone(), values });
        let values = |n: Node<'a, 'i>| -> Option<KeyValues> {
            Some(match n.name() {
                "ScalarInterpolator" => KeyValues::Float(n.field_or("keyValue", Vec::new()).ok()?),
                "PositionInterpolator" => KeyValues::Vec3(n.field_or("keyValue", Vec::new()).ok()?),
                "PositionInterpolator2D" => KeyValues::Vec2(n.field_or("keyValue", Vec::new()).ok()?),
                "ColorInterpolator" => KeyValues::Color(n.field_or("keyValue", Vec::new()).ok()?),
                "OrientationInterpolator" => KeyValues::Rotation(n.field_or("keyValue", Vec::new()).ok()?),
                "IntegerSequencer" => KeyValues::Int(n.field_or("keyValue", Vec::new()).ok()?),
                "BooleanSequencer" => KeyValues::Bool(n.field_or("keyValue", Vec::new()).ok()?),
                _ => return None,
            })
        };
        let users = self.users.get(&n.id()).cloned().unwrap_or_default();
        let node = if let Some(v) = values(n) {
            interp(v)
        } else {
            match n.name() {
                "TimeSensor" => AnimNode::Behaviour(Behaviour::TimeSensor {
                    cycle: f64::from(n.field_or("cycleInterval", 1.0f32).unwrap_or(1.0)),
                    looping: n.field_or("loop", false).unwrap_or(false),
                    start: n.field_or("startTime", 0.0f64).unwrap_or(0.0),
                    stop: n.field_or("stopTime", 0.0f64).unwrap_or(0.0),
                    enabled: n.field_or("enabled", true).unwrap_or(true),
                    active: false,
                    finished: f64::NAN,
                }),
                // Only a Viewpoint the scene drew becomes a source; the first is bound at load.
                "Viewpoint" | "OrthoViewpoint" => {
                    let index = *users.first()?;
                    AnimNode::Behaviour(Behaviour::Viewpoint { index, pending: index == 0 })
                }
                "BooleanFilter" => AnimNode::Behaviour(Behaviour::BooleanFilter),
                "BooleanToggle" => AnimNode::Behaviour(Behaviour::BooleanToggle {
                    state: n.field_or("toggle", false).unwrap_or(false),
                }),
                "Transform" => AnimNode::Target(Target::Transform(users)),
                "Material" | "UnlitMaterial" | "PhysicalMaterial" | "TwoSidedMaterial" => {
                    AnimNode::Target(Target::Material(users))
                }
                "DirectionalLight" | "PointLight" | "SpotLight" => AnimNode::Target(Target::Light(users)),
                "Switch" => AnimNode::Target(Target::Switch(users)),
                "ComposedShader" => {
                    let id = *self.shader_ids.get(&n.id())?;
                    AnimNode::Target(Target::Shader(self.scene.shaders[id as usize].fields.clone()))
                }
                "TextureTransform" => {
                    let (materials, fields) = self.texture_transforms.get(&n.id()).cloned()?;
                    AnimNode::Target(Target::TextureTransform { materials, fields })
                }
                _ => return None,
            }
        };
        let id = self.scene.animation.nodes.len() as u32;
        self.scene.animation.nodes.push(node);
        self.scene.animation.names.push(def.to_owned());
        self.anim_ids.insert(n.id(), id);
        Some(id)
    }

    fn finish(mut self) -> Scene {
        let s = &mut self.scene;
        let mut stats = s.report.stats;
        stats.shapes = s.draws.len();
        stats.unique_meshes = s.unique_meshes().len();
        for d in &s.draws {
            let n = d.mesh.primitive_count();
            match d.mesh.topology {
                Topology::Triangles => stats.triangles += n,
                Topology::Lines => stats.line_segments += n,
                Topology::Points => stats.points += n,
            }
        }
        stats.lights = s.lights.len();
        stats.viewpoints = s.viewpoints.len();
        stats.routes = s.animation.routes.len();
        s.report.stats = stats;
        self.scene
    }
}

/// The optional per-corner index lists of a polygon geometry.
/// 🦀 A *parameter struct*: named fields instead of a row of `&[i32]`
/// arguments, and `Default` (all empty) for geometries that have none.
#[derive(Debug, Clone, Copy, Default)]
struct Indices<'x> {
    color: &'x [i32],
    normal: &'x [i32],
    tex: &'x [i32],
}

/// `coordIndex` → faces of non-negative indices.
fn to_faces(index: &[i32]) -> Vec<Vec<u32>> {
    geometry::split_at_minus_one(index).into_iter().map(|f| f.into_iter().map(|i| i as u32).collect()).collect()
}

/// A Transform's fields (its matrix is [`Trs::matrix`]).
fn trs_of(n: Node<'_, '_>) -> Result<Trs> {
    Ok(Trs {
        translation: n.field_or("translation", Vec3::ZERO)?,
        rotation: n.field_or("rotation", Quat::IDENTITY)?,
        scale: n.field_or("scale", Vec3::ONE)?,
        scale_orientation: n.field_or("scaleOrientation", Quat::IDENTITY)?,
        center: n.field_or("center", Vec3::ZERO)?,
    })
}

/// The bytes of a `data:[mime];base64,…` url, or `None`.
fn data_url_bytes(url: &str) -> Option<Vec<u8>> {
    let rest = url.trim().strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    if !meta.ends_with(";base64") {
        return None;
    }
    crate::fields::base64_decode(payload)
}
