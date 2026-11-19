# Learning with x3d.wgpu

This project is meant to be read, broken and extended. The code carries two
kinds of signposts:

- **🎨** a graphics idea (pipelines, winding, normal matrices, MSAA resolve…)
- **🦀** a Rust idea (lifetimes, traits, iterators, error types, features…)

`grep -rn "🎨\|🦀" crates/` gives you a guided tour.

## A reading path

| Step | Read | You'll meet |
|---|---|---|
| 0 | `crates/x3d-render/examples/hello_triangle.rs` | the whole wgpu pipeline in ~150 lines |
| 1 | `x3d-scene/src/fields.rs` | traits + generics, const generics, `collect::<Result<…>>` |
| 2 | `x3d-scene/src/scene.rs` | `#[repr(C)]`, `cfg_attr` derives, `FromIterator`, `Arc` sharing |
| 3 | `x3d-scene/src/geometry.rs` | fan triangulation, Newell normals, creaseAngle, enums with data |
| 4 | `x3d-scene/src/parse.rs` | zero-copy parsing with lifetimes, extension traits, match guards |
| 5 | `x3d-render/src/gpu.rs` | instance → adapter → device, `async` + `pollster`, feature detection |
| 6 | `x3d-render/src/uniforms.rs` + `shaders/scene.wgsl` | WGSL layout rules, `bytemuck::Pod`, compile-time size checks |
| 7 | `x3d-render/src/renderer.rs` | bind groups, dynamic offsets, pipeline cache, MSAA, transparency sort |
| 8 | `x3d-render/src/camera.rs` | view/projection matrices, why wgpu's NDC is "directx" in glam |
| 9 | `x3d-render/src/headless.rs` | readback, 256-byte row alignment, `mpsc` channels |
| 9½ | `x3d-render/src/textures.rs` | mipmaps, anisotropy, bind groups per texture set, caching failures in an `Option` |
| 10 | `x3d-render/src/viewer.rs` | winit's `ApplicationHandler`, threads, `Arc<Mutex<…>>`, surfaces |
| 11 | `x3d-scene/src/runtime.rs` | an event cascade with a `VecDeque`, enums that carry state, slerp and HSV lerp |
| 12 | `x3d-render/src/glsl.rs` + `custom.rs` | translating a shader dialect, naga as a library (parse → validate → reflect) |
| 13 | `vs_wide` in `scene.wgsl` | instancing, clipping by hand, pixel-sized quads from clip space |
| 14 | `x3d-render/src/post.rs` + `shaders/post.wgsl` | float render targets, mip chains as blur, tone curves |
| 15 | `shadows.rs`, `sun_shadow`/`local_shadow` in the shader | depth-only passes, comparison samplers, cascades, texel snapping, cube faces from a 2D array |
| 16 | `budget.rs`, `gpu.rs` | limits and features as data; deciding once, explaining every decision |
| 17 | `timing.rs`, `bench.rs` | timestamp queries, async readback, `Arc<Mutex<…>>` state from a callback |
| 18 | `cull.rs` + `shaders/cull.wgsl` | frustum planes from a matrix, compute shaders, atomics, indirect draws |
| 19 | `conform.rs`, `tests/golden.rs`, `fuzz/` | receipts, tolerances with reasons, hashing inputs, fuzzing with libFuzzer |
| 20 | `environment.rs`, `environment()` in the shader | cube-map conventions, solid angles, spherical harmonics, importance sampling, half floats |

## Habits that make this fun

- **Keep the viewer open while you work.** `cargo run --release -- view scenes/01_primitives.x3d --shader crates/x3d-render/src/shaders/scene.wgsl` reloads the scene *and* the shader on every save.
- **Press 6** (facing view) whenever something looks wrong: red means you're seeing back faces, so the winding is reversed.
- **Press 2** (normals) to check lighting inputs before blaming the lighting.
- **`render --sheet`** shows all six views at once. Good for bug reports.
- **`RUST_LOG=info`** (or `wgpu_core=warn,x3d_render=debug`) for more detail.
- **Write the test first.** The GPU tests in `crates/x3d-render/tests/render.rs` check real pixels; copy one and change the expectation.

## Quests

Graded ★ (an evening) to ★★★★ (a project). Each says where to start and how to
know you're done. They're independent; pick what looks fun.

### 🎨 Graphics

**G1 ★ Play with hello_triangle.** Swap two vertices, add back-face culling, switch to `LineStrip`, change `SIZE` to 500 and watch the readback break (then fix it as `headless::read_texture` does).
*Done when* you can explain why 500 px breaks it and 512 doesn't.

**G2 ★★ Gradient skies.** `Background` already parses `skyColor`/`skyAngle`/`groundColor`/`groundAngle`. Draw a full-screen triangle before the scene whose fragment shader turns each pixel into a view ray (inverse view-projection), takes its angle from straight up, and blends the colours (X3D 24.4.1). 13 of the 92 x3d.demo scenes use a gradient.
*Done when* `report.approximated` no longer mentions Background gradients.

**G3 ★★ Reversed-Z.** Swap to `directx::perspective_infinite_reverse`, clear depth to 0.0, compare with `Greater`. Then zoom out on the terrain and compare z-fighting before and after.
*Why:* float precision is densest near 0; reversed-Z puts it where distance needs it.

**G4 ✅ Real GPU timing** — *done in 0.4* (`timing.rs`, `x3d-view bench`). **Follow-up ★★:** show the last frame's pass times in the viewer's title (they're already in `FrameStats::gpu`).

**G4b ★★★ One draw call.** GPU culling (`cull.rs`) decides visibility, but the CPU still issues one indirect call per draw. Merge all triangle meshes into one vertex/index buffer, move per-draw data into a storage buffer indexed by `instance_index` (`INDIRECT_FIRST_INSTANCE`), and replace the loop with one `multi_draw_indexed_indirect` per pipeline. Measure on Giza with `bench`.

**G5 ✅ Textures** — *done in 0.2, for Giza.* Read how: `Vertex::texcoord` (3D, so `TextureCoordinateGenerator` can store positions), `parse.rs` `texture`/`texcoords`/`texture_matrix`, `textures.rs`, and the top of `fs_main` (why every texture is sampled before any branch). **Follow-ups ★★:** generate mipmaps on the GPU in a compute shader, and do it gamma-correctly (the CPU version averages sRGB bytes, which darkens a little); honour `repeatS='false'` (it needs a second sampler); support `PixelTexture` (the Gobbler has one).

**G6 ✅ Fat points and lines** — *done.* Read `vs_wide`: one instance per segment (a point is a zero-length segment), six vertices per quad, expanded in clip space by the width in pixels, and clipped against the near plane *before* the divide. That last part also cured lavapipe's tile smears from lines passing behind the camera. **Follow-up ★★:** round caps and joins (`LineProperties.linetype` dashes, too).

**G7 ★★★ Instancing.** In `03_def_use_switch.x3d` 25 draws share one mesh. Group draws by (mesh, pipeline), put per-draw data in a *storage* buffer, and issue one `draw_indexed(.., 0..n)` per group, reading `@builtin(instance_index)` in the shader. Measure the draw-call count before and after.

**G8 ✅ PBR for PhysicalMaterial** — *done in 0.2.* Read `shade_pbr` in the shader and the `physical_material_matches_x_ite` test (78/255, as X_ITE measured).
**G8b ✅ Image-based lighting** — *done in 0.5* (`environment.rs`, `environment()` in the shader, [RESEARCH-ibl.md](RESEARCH-ibl.md)). Try `--ibl single` on `09_environment.x3d` and watch rough gold go dark: that is the energy a single GGX bounce loses. **Follow-ups:** ★★ prefilter on the GPU in a compute shader (it's CPU now: fine for 64², slow for real HDRIs); ★★ `ImageCubeMapTexture` (one image, cross layout) and HDR (`.hdr`/`.exr`) cubes; ★★★ compare a *non-uniform* environment with X_ITE (the furnace can't see face orientation).

**G8c ✅ Tone mapping and bloom** — *done.* Read `post.rs`. **Follow-ups ★★:** auto-exposure (average log-luminance from the smallest bloom mip, eased over time); a lens-dirt texture on the bloom; compare Giza's disco under ACES and AgX.

**G8d ★★★ Gaussian splats.** Giza's Khufu has splat levels (`khufu-g*-splats-*.xml`, not in the repo) and Pantheon is 50,058 splats. Sort splats by depth each frame and draw them as instanced, alpha-blended quads.

**G9 ✅ Shadows** — *done in 0.4:* texel-stable sun cascades, spot (perspective) and point (six faces) maps, X3D `shadows`/`shadowIntensity`. **Follow-ups:** ★★ blend between cascades at their seams; ★★ swap 3×3 PCF for Poisson disc or PCSS (soft shadows that harden near the caster); ★★ cull casters per cascade.

**G10 ★★★ Picking.** Render draw indices into an `R32Uint` target, read back the pixel under the cursor on click, and print the shape's `label` (its X3D node name and DEF).

**G12 ★★ Fog.** Implement X3D `Fog` and `LocalFog` (LINEAR and EXPONENTIAL) in the fragment shader. The Osiris Shaft has a `LocalFog` waiting for it.

**G11 ★★ Billboards, properly.** Keep Billboard as its own node in the scene and rebuild its matrix each frame from the camera (around `axisOfRotation`, or screen-aligned when it's `0 0 0`).

### 🦀 Rust

**R1 ✅ Inline** — *done in 0.2.* Read `Loader::inline` and `InlineState`: a `&mut` lent down through each nested load (a reborrow), a stack for cycles, an `Arc<Scene>` cache so a file Inlined twice is parsed once, and `Scene::splice` renumbering LOD groups and light scopes. **Follow-up ★★:** `IMPORT`/`EXPORT` (Giza uses them to wire animations across files); they matter once there are ROUTEs to run.

**R1b ✅ Animation** — *done.* Read `runtime.rs`: `Scene::advance(t)` ticks every TimeSensor, pushes events into a queue, and follows ROUTEs, firing each route at most once per timestamp (X3D's loop breaking). `Changes` tells the renderer what to re-upload. **Follow-ups:** ★★ TouchSensor + picking (G10) so the Gobbler reacts to clicks; ★★★ a `Script` node with a tiny expression language; ★★★ ParticleSystem on the GPU with a compute shader.

**R1c ✅ ComposedShader** — *done.* Read `glsl.rs` then `custom.rs`. **Follow-up ★★:** the `shaders --show` output is GLSL 450; ask naga to write it back out as WGSL and read what your shader became.

**R2 ★★★ Ear clipping.** `convex='false'` faces are fan-triangulated (wrong for concave polygons). Project each face onto its dominant plane, ear-clip it, and add a proptest: for random simple polygons, the triangle areas must sum to the polygon's area.

**R3 ★★ Pretty errors.** Bring in `miette` and make `Error::Field` show the offending line of XML with the bad value underlined. roxmltree already gives you byte ranges.

**R4 ★★ Data parallelism.** Tessellation is independent per geometry. Collect the geometry nodes first, then tessellate with `rayon`'s `par_iter`. What has to change about `Loader` for that to compile? (Hint: `&mut self` and `Send`.)

**R5 ★★ Benchmarks.** Add `criterion` and benchmark `Polygons::tessellate` and `Mesh::weld` on a 512 × 512 ElevationGrid. Then try `ahash` or `FxHashMap` in `weld` and measure.

**R6 ★★ A real watcher.** Swap the polling thread for the `notify` crate with debouncing. Keep the `EventLoopProxy` design.

**R7 ★★★ An egui panel.** Add `egui-wgpu` + `egui-winit` (match their wgpu version!) and show the settings, the loss report and a scene tree with per-shape visibility toggles.

**R8 ★★★★ In the browser.** wgpu runs on WebGPU. Build the viewer for `wasm32-unknown-unknown` with `trunk`. What has to go behind `#[cfg(not(target_arch = "wasm32"))]`? (The watcher thread and `pollster`, for a start.)

**R9 ★★★★ ClassicVRML.** Write a `.x3dv` parser with `winnow` that builds the same scene. The `Loader` walks roxmltree nodes, so you'll want a small trait over "a node with fields and children" that both encodings implement.

**R10 ★ Make the lints stricter.** Turn on `clippy::pedantic` in `Cargo.toml` and fix (or consciously `#[allow]`) what it finds.

### 🧩 X3D fidelity (tying back to x3d.lab)

**X1 ★ A diff command.** *The metric and the heat map exist* (`diff.rs`, used by the golden tests); wire them to the CLI. Add `x3d-view diff a.png b.png` that prints the mean |Δ| per pixel (0–255), the metric the x3d.lab findings use, and writes a heat-map PNG.

**X2 ★★★ Against X_ITE.** Render the same x3d.demo scenes with X_ITE (headless Chrome) and with `x3d-view render`, then diff them. Which nodes cause the biggest gaps? Each one is a quest of its own.

**X3 ★★ Point and spot falloff.** x3d.lab found that glTF's 1/d² can't match X3D's default `attenuation='1 0 0'` (no falloff). This renderer implements X3D's formula directly. Build a scene that shows the difference, and add a pixel test.

## Glossary

- **Adapter / device / queue**: a GPU / your connection to it / where you submit work.
- **Bind group**: a set of resources (buffers, textures) bound together; the shader sees them as `@group(n) @binding(m)`.
- **Dynamic offset**: binding a different slice of one buffer per draw without a new bind group.
- **Pipeline**: compiled shaders plus fixed-function state (topology, culling, blending, depth test).
- **MSAA resolve**: averaging the multisampled image into a normal one at the end of a pass.
- **NDC**: normalized device coordinates; in wgpu x, y ∈ [−1, 1] with y up, and depth ∈ [0, 1].
- **Winding**: the order of a triangle's corners on screen; CCW = front face here (X3D `ccw='true'`).
- **Normal matrix**: the inverse-transpose of the model matrix; keeps normals perpendicular under non-uniform scale.
- **Crease angle**: X3D's threshold below which neighbouring faces share a smooth normal.
