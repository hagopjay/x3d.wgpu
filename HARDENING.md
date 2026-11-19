# Hardening x3d.wgpu 0.4: receipts

Each number below names its source: the adapter and driver that produced it, the tool that measured it, or the validator that checked it. Unless a row says otherwise, everything was measured on 19 Nov 2026 on one machine:

| | |
|---|---|
| **Adapter** | llvmpipe (LLVM 20.1.2, 256 bits), Vulkan, device type Cpu |
| **Driver** | Mesa 25.2.8-0ubuntu0.24.04.5 (lavapipe) |
| **Rust** | rustc 1.97.0 (stable) |
| **Machine** | 4 vCPU |

No hardware GPU was available, so nothing here is a hardware-GPU number. Grades follow the Loom: **C** means a named tool ran and recorded the value; **L** means read or reasoned.

## 1. GPU tests that cannot pass by skipping

| What | Where | Grade |
|---|---|---|
| With `X3D_REQUIRE_GPU=1` (CI sets it), "no adapter" panics instead of printing SKIPPED | `tests/render.rs`, `tests/golden.rs` and `tests/conformance.rs`: each has its own `gpu()` helper | C |
| Each GPU test binary names the adapter it ran on | the `GPU tests on: …` line in the test output | C |
| Golden images: 8 sample scenes plus the still life with shadows, ACES and bloom, at 192×108 with MSAA 4 | `tests/golden/*.png`, blessed on the adapter above (`tests/golden/SOURCE.txt`) | C |
| Tolerance per adapter, with the reason for each row | `tests/golden/TOLERANCE`: llvmpipe 2/0.5‰ · swiftshader 6/5‰ · `*` 12/15‰ | L (the `*` row is reasoned from Loom LEARNINGS-shader-vulkan finding 5: max 14/255 between two LLVM builds) |
| Negative control: blacking out 10×10 pixels of the 06_terrain golden makes the test fail | report: mean \|Δ\| 0.9211, max 242, 100 bad pixels, 4.823‰ → FAIL | C |
| On failure the test writes `target/golden/<name>.got.png`, a heat map and `report.txt`; CI uploads them | `.github/workflows/ci.yml` | L |
| A self-hosted `gpu` job runs the same suite on a real GPU | `ci.yml`, enabled by the repository variable `X3D_GPU_RUNNER=true` | L (not run: there is no such runner yet) |

Test count: **109 pass, 0 fail** with `X3D_REQUIRE_GPU=1` on the adapter above (91 in 0.3).

## 2. Robustness

| Defence | How | Evidence |
|---|---|---|
| Fuzzing the loader and runtime | `fuzz/fuzz_targets/parse.rs`: any bytes → `parse_str_with` → `advance` → `bind_viewpoint` → `bounds` | see the fuzz table below |
| Fuzzing shader text, which is untrusted input | `fuzz/fuzz_targets/glsl.rs`: vertex 0xFF fragment flag → `glsl::translate` → naga parse, validate and reflect | see the fuzz table below |
| Hostile scenes render without panics or wgpu validation errors | 9 cases (NaN and ±inf coordinates, out-of-range indices, empty geometry, 1e38 scales, zero scale, degenerate lights, 20 point lights asking for shadows, a broken shader, a ROUTE loop), each checked with an error scope | test `hostile_scenes_render_without_panics_or_validation_errors` (C) |
| A validation error is logged, not fatal | `on_uncaptured_error` logs; the test makes an out-of-bounds copy and then renders correctly | test `an_uncaptured_validation_error_is_logged_not_fatal` (C) |
| Device lost | `set_device_lost_callback` sets `Gpu::lost`; the viewer drops all GPU state and rebuilds it on a fresh adapter | test `a_lost_device_is_noticed_not_crashed` uses `device.destroy()` (C). The viewer rebuild is L: it was not exercised with a real driver reset |
| Surface errors | Outdated or Lost: reconfigure and redraw. Timeout or Occluded: skip and redraw. A surface with no formats returns an error instead of panicking on `formats[0]` | `viewer.rs` (L) |

**Fuzz runs.** None found a crash. Rigs: cargo-fuzz 0.13.2, libFuzzer, `-s none` (no sanitizer, since the code is safe Rust), stable rustc 1.97.0 with `RUSTC_BOOTSTRAP=1` (the sandbox could not download nightly). Each run was 301 s on 1 core.

| Target | Seeds | Dictionary | Runs | exec/s | Coverage (edges) | Features | Corpus | Crashes |
|---|---|---|---|---|---|---|---|---|
| parse | 8 sample scenes | none | 7,532,419 | 25,024 | 3,204 | 12,386 | 3,178 | 0 |
| parse (2nd run, same corpus) | the run-1 corpus | `fuzz/dict/x3d.dict` (265 tokens taken from the parser's own strings) | 7,475,816 | 24,836 | 3,365 | 13,967 | 3,217 | 0 |
| glsl | 2 synthetic shaders (sprite and plain) | none | 2,195,654 | 7,294 | 5,699 | 20,358 | 3,242 | 0 |

CI fuzzes each target for 60 s on every push and for 15 min weekly, and uploads any crash it finds.

## 3. Closer to the metal

**The budget.** `x3d-view adapters --budget` prints every setting and the limit or feature it came from:

```text
decision                   value        source
tier                       Reduced      profile Auto, device_type Cpu (llvmpipe (LLVM 20.1.2, 256 bits))
sun shadow map             2048²        tier Reduced wants 2048; limits.max_texture_dimension_2d = 16384
cascades                   4            tier Reduced; limits.max_texture_array_layers = 2048
spot/point shadow layers   16 × 512²    tier Reduced (a spot light takes 1 layer, a point light 6); limits.max_texture_array_layers = 2048
shadow memory              80 MiB       Depth32Float = 4 bytes per texel, sum over all layers
max MSAA                   4            texture-format features MULTISAMPLE_X4 of Rgba8Unorm and Depth32Float
HDR (tone map, bloom)      on           Rgba16Float renderable+filterable = true; tier Reduced (Minimal emulates WebGL2: off)
GPU timestamps             on           feature TIMESTAMP_QUERY = true
pipeline cache             on           feature PIPELINE_CACHE = true
GPU frustum culling        on           limits.max_compute_workgroup_size_x = 1024 (need 64), max_storage_buffers_per_shader_stage = 48 (need 2)
```

```text
decision                   value        source
tier                       Minimal      profile Weak, device_type Cpu (llvmpipe (LLVM 20.1.2, 256 bits))
sun shadow map             1024²        tier Minimal wants 1024; limits.max_texture_dimension_2d = 16384
cascades                   1            tier Minimal; limits.max_texture_array_layers = 256
spot/point shadow layers   0 × 0²       tier Minimal (a spot light takes 1 layer, a point light 6); limits.max_texture_array_layers = 256
shadow memory              4 MiB        Depth32Float = 4 bytes per texel, sum over all layers
max MSAA                   4            texture-format features MULTISAMPLE_X4 of Rgba8Unorm and Depth32Float
HDR (tone map, bloom)      off          Rgba16Float renderable+filterable = true; tier Minimal (Minimal emulates WebGL2: off)
GPU timestamps             off          feature TIMESTAMP_QUERY = false
pipeline cache             off          feature PIPELINE_CACHE = false
GPU frustum culling        off          limits.max_compute_workgroup_size_x = 0 (need 64), max_storage_buffers_per_shader_stage = 0 (need 2)
```

**Per-pass GPU time** comes from timestamp queries: `x3d-view bench`, the still life at 640×360 with MSAA 4, shadows, ACES, bloom 0.15 and GPU culling, 30 frames, on an idle machine:

```text
scenes/05_still_life.x3d at 640×360, msaa 4, Gpu
adapter: llvmpipe (LLVM 20.1.2, 256 bits) — Vulkan, Cpu, driver llvmpipe Mesa 25.2.8-0ubuntu0.24.04.5 (LLVM 20.1.2)
pass        frames   mean ms    p50 ms    p95 ms    max ms  source
shadow          30    21.904    20.292    30.192    53.312  GPU, timestamp queries, 1 ns/tick
cull            30     0.067     0.051     0.160     0.226  GPU, timestamp queries, 1 ns/tick
scene           30    24.947    24.104    31.659    38.046  GPU, timestamp queries, 1 ns/tick
bloom           30     6.313     6.422     7.216     8.630  GPU, timestamp queries, 1 ns/tick
tone map        30     3.902     3.829     4.393     8.833  GPU, timestamp queries, 1 ns/tick
cpu             30     0.500     0.460     0.652     0.670  CPU record+submit, std::time::Instant
set_scene 5.4 ms · first frame (pipelines) 285.7 ms · Instant
pipeline cache: off (no PIPELINE_CACHE, or X3D_PIPELINE_CACHE=off)
draws: 12 in scene · 12 issued · 0 CPU-culled · GPU kept 12 · 4 shadow passes (last frame)
```

The same run with `--profile weak` (an emulated WebGL2-class GPU): no timestamps, so every GPU column is null and none is filled with a CPU number. The CPU column includes queue backpressure here, because nothing waits on the GPU between frames.

```text
scenes/05_still_life.x3d at 640×360, msaa 4, Gpu
adapter: llvmpipe (LLVM 20.1.2, 256 bits) — Vulkan, Cpu, driver llvmpipe Mesa 25.2.8-0ubuntu0.24.04.5 (LLVM 20.1.2)
pass        frames   mean ms    p50 ms    p95 ms    max ms  source
shadow           0      null      null      null      null  GPU, null: no TIMESTAMP_QUERY on this device (or pass not run)
cull             0      null      null      null      null  GPU, null: no TIMESTAMP_QUERY on this device (or pass not run)
scene            0      null      null      null      null  GPU, null: no TIMESTAMP_QUERY on this device (or pass not run)
bloom            0      null      null      null      null  GPU, null: no TIMESTAMP_QUERY on this device (or pass not run)
tone map         0      null      null      null      null  GPU, null: no TIMESTAMP_QUERY on this device (or pass not run)
cpu             30    22.689    22.829    29.803    30.466  CPU record+submit, std::time::Instant
set_scene 3.4 ms · first frame (pipelines) 175.4 ms · Instant
pipeline cache: off (no PIPELINE_CACHE, or X3D_PIPELINE_CACHE=off)
draws: 12 in scene · 12 issued · 0 CPU-culled · GPU kept n/a · 1 shadow passes (last frame)
```

**Culling at Giza scale.** Giza v2.1 (private assets; numbers only) via `giza-tour.x3d --load-all-inlines --viewpoint 1`, 640×360, 10 frames:

| Cull mode | Draws issued | Culled | Scene pass, GPU mean ms | Cull pass ms |
|---|---|---|---|---|
| off | 167 | 0 | 227.7 | — |
| cpu | 49 | 118 (CPU) | 222.6 | — |
| gpu | 167 (indirect) | 118 (GPU kept 49) | 215.6 | 0.055 |
| cpu + `--shadows` (4 cascades) | 49 | 118 | 225.0 | shadow pass 177.9 |

Source: `x3d-view bench`, timestamp queries, on the adapter above. **Finding:** on lavapipe, culling 71 % of the draws saves only about 3–5 % of the scene pass. A CPU rasteriser spends its time on fragments, not on draws whose triangles get clipped. The draw-call savings should matter much more on a hardware GPU, but that has not been measured (S).

**Pipeline cache.** Time to the first frame of the still life (all pipelines built), 320×180, shadows + ACES + bloom:

| Mesa disk cache | wgpu `PipelineCache` | First frame |
|---|---|---|
| off (`MESA_SHADER_CACHE_DISABLE=true`) | off | 1981.5 ms |
| off | warm | 1784.7 ms |
| warm | off | 381.9 ms |
| warm | warm | 293.9 ms |

Source: `bench`'s first-frame time (`std::time::Instant`), one run per row. **Finding:** lavapipe returns a 96-byte `PipelineCache` blob, header only, so the wgpu cache's effect here is within noise (L). Mesa's own disk cache does the real work. Hardware drivers (NVIDIA, RADV, ANV) fill `VkPipelineCache` with compiled code, but that is unmeasured (S). This is also the one `unsafe` call in the project: `pipeline_cache.rs` says why it is sound.

**Reuse.** Per frame, the renderer creates only a command encoder. Buffers, bind groups and the 16-byte shadow-pass index slots are created at start-up or per scene, and the indirect-args and sphere buffers per scene (L, from reading the render path). Host allocations per frame were not measured.

**Not done:** merging meshes into one buffer so that `multi_draw_indexed_indirect` replaces per-draw indirect calls. GPU culling today decides visibility, but the CPU still issues one call per draw.

## 4. Shadows

| Feature | How | Test |
|---|---|---|
| Sun cascades | 1–4 slices with the practical split (λ 0.7), one orthographic map each, the cascade picked by view depth | `splits_grow_and_end_at_far`, `the_whole_slice_is_inside_its_cascade` (unit, C) |
| Texel-stable fit (no shimmer) | the radius comes from a bounding sphere rounded up to 1/16; the centre is snapped to whole texels in a light space without translation | `cascade_boxes_move_in_whole_texels`: 4 sub-texel camera shifts, every world point stays on the same sub-texel position within 0.01 texel (C) |
| Spot shadows | one perspective layer, field of view = 2 × cutOffAngle | `a_spot_light_with_shadows_true_casts_a_perspective_shadow` (GPU, C) |
| Point shadows | six 92° layers, the face picked by major axis | `a_point_light_casts_into_its_cube_faces` (GPU, C) |
| X3D 4 `shadows` and `shadowIntensity` | lights with `shadows='true'` cast; with none, the first DirectionalLight does (0.3's behaviour) | `shadow_intensity_scales_the_darkening` (C) |
| Layer budget | sun 4096/2048/1024 by tier; local lights 16 layers at 1024/512/0; lights over budget are logged and drawn unshadowed | `20 point lights asking for shadows` hostile case (C) |
| Sky domes don't shadow the world | geometry that encloses both the camera and over 90 % of the scene doesn't cast | L: Giza's sky bubble shadowed everything before this rule (seen in a render, not a test) |

## 5. Weak GPUs

`--profile weak` emulates a WebGL2-class device: `Limits::downlevel_webgl2_defaults()` and no optional features. That is what a GPU Chrome blocks for WebGPU (the owner's UHD 620) falls back to. Results:

- HDR, tone mapping and bloom turn off, with one warning.
- GPU culling falls back to CPU culling.
- Shadows run with one 1024² cascade and no spot or point shadows.

Test `a_weak_gpu_profile_degrades_instead_of_failing`: the same picture with and without the extras requested, and the shadow is still present (C, on lavapipe with weak limits). **Not measured on a UHD 620.** Native Vulkan on that GPU (Mesa ANV) would get the Auto profile; only WebGPU-in-Chrome is blocklisted.

## 6. Conformance receipts

`x3d-view conform conformance --receipt receipt.json` renders scenes whose expected pixels come from **X_ITE 16.7.1 measurements recorded by the Loom**: `hagopjay/x3d.loom@9a73e44:runs/shader-vulkan-20261117/xite-furnace-colour.json`, a probe in Chromium 153 on SwiftShader. The scenes were rebuilt node for node from its probe script.

| Row | Expected (X_ITE) | x3d-view | \|Δ\| | Tolerance and why | Verdict |
|---|---|---|---|---|---|
| exp2_a_unlit_default | 128 | 128 | 0 | 1: 8-bit rounding | pass |
| exp2_b_legacy_default | 128 | 128 | 0 | 1: 8-bit rounding | pass |
| exp2_c_physical_default | 186 | 188 | 2 | 2: X_ITE encodes with pow(1/2.2), we use piecewise sRGB (Loom finding 1) | pass |
| ctrl_noenv_m0_r1_c0.5 (sphere / sky) | 0 / 255 | 0 / 255 | 0 / 0 | 1: exact | pass |
| ctrl_noenv_m1_r0_c1 | 0 | 0 | 0 | 1: exact | pass |
| furnace_* (9 rows) | 187–255 | — | — | needs EnvironmentLight (not implemented) | not run |

**Validity.** Every scene was checked by **x3d_mcp@39c6e94** (`validate_xml` + `semantic_diagnostics`), run by the Loom's grammar and meaning witnesses (`kit/loom grammar`, x3d.loom@9a73e44). `conformance/validators/MANIFEST.json` records each file's sha256 and verdict. A row whose scene changed since validation is refused as `stale validation`, which the test `a_scene_edited_after_validation_is_refused` checks. `conformance/validate.sh` re-runs the validators.

**What x3d_mcp found in our own sample scenes:** 9 semantic errors. Each scene declared `profile='Interchange'` but used nodes outside it:
- Switch;
- Point and SpotLight;
- PhysicalMaterial;
- ElevationGrid;
- QuadSet;
- Billboard;
- Inline;
- Text.

0.4 adds the missing `<component>` declarations. Now all 13 files are XSD 4.0 valid, with 0 errors and 0 warnings (C).

## 7. What this does not show

- Any hardware GPU, any driver other than lavapipe, any OS other than Linux.
- The UHD 620 itself.
- Viewer recovery from a real device loss (only `destroy()` was tested).
- Fuzzing with AddressSanitizer: memory safety comes from safe Rust, except the one cache call.
- The X_ITE numbers were not re-measured here; they are the Loom's record.
