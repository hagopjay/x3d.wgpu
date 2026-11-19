//! Inline and textures: features that need real files next to each other.

use std::path::{Path, PathBuf};

use glam::{Mat4, Vec3};
use x3d_scene::{LoadOptions, load, load_with};

/// A throwaway folder that deletes itself.
/// 🦀 `Drop` runs when the value goes out of scope — even if the test panics.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("x3d-scene-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn write(&self, name: &str, body: &str) -> PathBuf {
        let path = self.0.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, format!("<X3D><Scene>{body}</Scene></X3D>")).unwrap();
        path
    }
    fn touch(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not really an image").unwrap();
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn origin_of(m: &Mat4) -> Vec3 {
    m.transform_point3(Vec3::ZERO)
}

#[test]
fn inline_places_the_other_file_under_the_transform() {
    let dir = TempDir::new("inline");
    dir.write(
        "parts/ball.x3d",
        "<Transform translation='0 1 0'><Shape><Sphere/></Shape></Transform>
         <PointLight location='0 0 0'/>",
    );
    let main = dir.write(
        "main.x3d",
        "<Transform translation='10 0 0'><Inline url='\"missing.x3d\" \"parts/ball.x3d\"'/></Transform>
         <Transform translation='-10 0 0'><Inline url='\"parts/ball.x3d\"'/></Transform>",
    );
    let s = load(&main).unwrap();
    let origins: Vec<Vec3> = s.draws.iter().map(|d| origin_of(&d.transform)).collect();
    assert_eq!(origins, [Vec3::new(10.0, 1.0, 0.0), Vec3::new(-10.0, 1.0, 0.0)]);
    // Same file twice → parsed once → the mesh is shared.
    assert!(std::sync::Arc::ptr_eq(&s.draws[0].mesh, &s.draws[1].mesh));
    assert_eq!(s.lights.len(), 2, "the inlined light comes along, moved");
    assert_eq!(s.report.inlined, ["ball.x3d", "ball.x3d"]);
}

#[test]
fn inline_cycles_and_load_false() {
    let dir = TempDir::new("cycle");
    dir.write("a.x3d", "<Shape><Box/></Shape><Inline url='\"b.x3d\"'/>");
    dir.write("b.x3d", "<Shape><Cone/></Shape><Inline url='\"a.x3d\"'/>");
    let lazy = dir.write("lazy.x3d", "<Inline load='false' url='\"a.x3d\"'/>");

    let s = load(dir.0.join("a.x3d")).unwrap();
    assert_eq!(s.draws.len(), 2, "a's box and b's cone, then the cycle stops");
    assert!(s.report.warnings.iter().any(|w| w.contains("cycle")), "{:?}", s.report.warnings);

    assert_eq!(load(&lazy).unwrap().draws.len(), 0, "load='false' is honoured");
    let eager = load_with(&lazy, &LoadOptions { load_all_inlines: true, ..Default::default() }).unwrap();
    assert_eq!(eager.draws.len(), 2);
}

#[test]
fn physical_material_texture_slots_follow_container_field() {
    let dir = TempDir::new("textures");
    let albedo = dir.touch("tex/albedo.jpg");
    let orm = dir.touch("tex/orm.jpg");
    let main = dir.write(
        "main.x3d",
        "<Shape>
           <Appearance>
             <PhysicalMaterial baseColor='0.9 0.8 0.7' metallic='0' roughness='0.9' normalScale='0.5'>
               <ImageTexture containerField='baseTexture' url='\"tex/albedo.jpg\"'/>
               <ImageTexture containerField='metallicRoughnessTexture' DEF='Orm' url='\"tex/orm.jpg\"'/>
               <ImageTexture containerField='occlusionTexture' USE='Orm'/>
             </PhysicalMaterial>
           </Appearance>
           <Box/>
         </Shape>",
    );
    let s = load(&main).unwrap();
    let m = &s.draws[0].material;
    let pbr = m.pbr.expect("PhysicalMaterial → pbr");
    assert_eq!((pbr.metallic, pbr.roughness, pbr.normal_scale), (0.0, 0.9, 0.5));
    let path = |t: &Option<x3d_scene::TextureRef>| t.as_ref().map(|t| t.path.clone());
    let canon = |p: &Path| Some(p.canonicalize().unwrap());
    assert_eq!(path(&m.textures.base), canon(&albedo));
    assert_eq!(path(&m.textures.metallic_roughness), canon(&orm));
    // The USE element carries its own containerField: same file, other slot.
    assert_eq!(path(&m.textures.occlusion), canon(&orm));
    assert!(m.textures.normal.is_none());
}

#[test]
fn coord_texture_generation_goes_through_the_column_major_matrix() {
    let dir = TempDir::new("texgen");
    // u = x/8 + 2, v = z/8 + 3 — like the Giza terrain.
    let main = dir.write(
        "main.x3d",
        "<Shape>
           <Appearance><PhysicalMaterial/>
             <TextureTransformMatrix3D matrix='0.125 0 0 0  0 0 0 0  0 0.125 0 0  2 3 0 1'/>
           </Appearance>
           <IndexedTriangleSet index='0 1 2'>
             <TextureCoordinateGenerator mode='COORD'/>
             <Coordinate point='0 0 0  8 0 0  0 0 8'/>
           </IndexedTriangleSet>
         </Shape>",
    );
    let s = load(&main).unwrap();
    let d = &s.draws[0];
    let st: Vec<[f32; 2]> = d
        .mesh
        .vertices
        .iter()
        .map(|v| {
            let t = d.material.texture_matrix * Vec3::from(v.texcoord).extend(1.0);
            [t.x, t.y]
        })
        .collect();
    assert!(st.contains(&[2.0, 3.0]) && st.contains(&[3.0, 3.0]) && st.contains(&[2.0, 4.0]), "{st:?}");
}

#[test]
fn texture_transform_2d_matches_the_x3d_formula() {
    // translation 0.5 then scale 2 about the centre (0.5, 0.5):
    // Tc' = −C·S·R·C·T·Tc → (0, 0) ↦ ((0 + 0.5 + 0.5)·2 − 0.5, …) = (1.5, 1.5)
    let dir = TempDir::new("tt");
    let main = dir.write(
        "main.x3d",
        "<Shape><Appearance><Material/>
           <TextureTransform translation='0.5 0.5' scale='2 2' center='0.5 0.5'/>
         </Appearance><Box/></Shape>",
    );
    let m = load(&main).unwrap().draws[0].material.texture_matrix;
    assert_eq!(m.transform_point3(Vec3::ZERO), Vec3::new(1.5, 1.5, 0.0));
}

#[test]
fn invisible_things_are_not_drawn() {
    let dir = TempDir::new("visible");
    let main = dir.write(
        "main.x3d",
        "<Shape visible='false'><Box/></Shape>
         <Group visible='false'><Shape><Sphere/></Shape></Group>
         <Shape><Cone/></Shape>",
    );
    let s = load(&main).unwrap();
    assert_eq!(s.draws.len(), 1);
    assert!(s.draws[0].mesh.label.starts_with("Cone"));
}

#[test]
fn default_texture_mapping_uses_the_two_longest_sides() {
    // A 4 × 2 quad in the XY plane: s runs along x (0..1), t along y (0..0.5).
    let s = x3d_scene::parse_str(
        "<X3D><Scene><Shape><IndexedFaceSet coordIndex='0 1 2 3'>
           <Coordinate point='0 0 0  4 0 0  4 2 0  0 2 0'/></IndexedFaceSet></Shape></Scene></X3D>",
    )
    .unwrap();
    let mut st: Vec<[f32; 3]> = s.draws[0].mesh.vertices.iter().map(|v| v.texcoord).collect();
    st.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(st, [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0]]);
}

#[test]
fn lod_keeps_every_level_and_tags_it() {
    // Like Giza's Khufu: ranges so small that any real distance picks the last level.
    let s = x3d_scene::parse_str(
        "<X3D><Scene><LOD center='0 70 0' range='1e-06 2e-06'>
           <Group/><Group/><Shape><Box/></Shape>
         </LOD></Scene></X3D>",
    )
    .unwrap();
    assert_eq!(s.draws.len(), 1);
    let tag = &s.draws[0].lod[0];
    assert_eq!((tag.level, tag.center), (2, Vec3::new(0.0, 70.0, 0.0)));
    assert!(tag.active(Vec3::new(0.0, 0.0, 500.0)), "far away → level 2 is shown");
}

#[test]
fn scoped_lights_only_reach_their_group() {
    let dir = TempDir::new("scopes");
    dir.write(
        "shaft.x3d",
        "<Group><PointLight DEF='Torch' global='false' intensity='30'/><Shape><Box/></Shape></Group>",
    );
    let main = dir.write(
        "main.x3d",
        "<DirectionalLight/>
         <Shape><Sphere/></Shape>
         <Transform translation='5 0 0'><Inline url='\"shaft.x3d\"'/></Transform>",
    );
    let s = load(&main).unwrap();
    let (sun, torch) = (&s.lights[0], &s.lights[1]);
    let (sphere, shaft_box) = (&s.draws[0], &s.draws[1]);
    // The root's DirectionalLight (global='false' by default) is scoped to
    // the root group, which holds everything, inlined files included.
    assert!(sun.reaches(&sphere.scopes) && sun.reaches(&shaft_box.scopes));
    // The torch lights only its own group inside the inlined file.
    assert!(torch.reaches(&shaft_box.scopes));
    assert!(!torch.reaches(&sphere.scopes), "the torch must not light the sphere");
}
