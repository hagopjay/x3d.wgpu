# Research: EnvironmentLight shading vs X_ITE (x3d.wgpu 0.5)

**Question:** which image-based-lighting model reproduces X_ITE's measured PhysicalMaterial pixels under X3D 4.1 `EnvironmentLight`? And what is left over once it does?

**Reference data (C):** X_ITE 16.7.1 in Chromium 153.0.8010.12 (Playwright 1.63.0, ANGLE on SwiftShader).
- Record: `hagopjay/x3d.loom@9a73e44:runs/shader-vulkan-20261117/xite-furnace-colour.json`.
- Probe: centre 5×5 mean of a 128×128 canvas, `toneMapping NONE`.
- Scene: one sphere (r 1.2, camera at z 5, fov 0.7854), headlight off, white Background, and an EnvironmentLight whose diffuse and specular cubes are six 1×1 white PNGs.
- Our scenes (`conformance/scenes/*.x3d`) rebuild the probe's scenes node for node. All validate with **x3d_mcp@39c6e94** (XSD 4.1, 0 semantic errors).

**Our side (C):** `x3d-view render --size 128x128 --msaa 1`, same probe, on llvmpipe (LLVM 20.1.2), Mesa 25.2.8, Vulkan. Four variants:

| Model | What it does |
|---|---|
| single | diffuse·E/π + prefiltered·(F0·A + B). Split sum, with Karis's analytic DFG fit. |
| split | Diffuse scaled by (1 − F0·A − B). |
| multi | Fdez-Agüera 2019 multiple-scattering compensation. **Default in 0.5.** |
| multi γ2.2 | `multi` with `--encode gamma22`: linear → display by pow(1/2.2), X_ITE's curve (x_ite.js 16.7.1 l.67921), instead of piecewise sRGB. |

## Results (bytes, red channel; all three channels agree)

| Row | Intensity | Metallic | Roughness | Base | X_ITE | single | split | multi | multi γ2.2 |
|---|---|---|---|---|---|---|---|---|---|
| ctrl_intensity05_m0_r1_c0.5 | 0.5 | 0 | 1 | 0.5 | **137** | 139 | 138 | 138 | 137 |
| ctrl_intensity05_m1_r0.5_c1 | 0.5 | 1 | 0.5 | 1 | **186** | 162 | 162 | 188 | 186 |
| ctrl_intensity05_m1_r0_c1 | 0.5 | 1 | 0 | 1 | **186** | 188 | 188 | 188 | 186 |
| ctrl_intensity0_m0_r1_c0.5 | 0 | 0 | 1 | 0.5 | **0** | 0 | 0 | 0 | 0 |
| ctrl_intensity0_m1_r0.5_c1 | 0 | 1 | 0.5 | 1 | **0** | 0 | 0 | 0 | 0 |
| ctrl_intensity0_m1_r0_c1 | 0 | 1 | 0 | 1 | **0** | 0 | 0 | 0 | 0 |
| ctrl_noenv_m0_r1_c0.5 | — | 0 | 1 | 0.5 | **0** | 0 | 0 | 0 | 0 |
| ctrl_noenv_m1_r0.5_c1 | — | 1 | 0.5 | 1 | **0** | 0 | 0 | 0 | 0 |
| ctrl_noenv_m1_r0_c1 | — | 1 | 0 | 1 | **0** | 0 | 0 | 0 | 0 |
| furnace_m0_r0.5_c0.5 | 1 | 0 | 0.5 | 0.5 | **189** | 193 | 190 | 190 | 189 |
| furnace_m0_r0_c0.5 | 1 | 0 | 0 | 0.5 | **189** | 195 | 191 | 191 | 190 |
| furnace_m0_r1_c0.5 | 1 | 0 | 1 | 0.5 | **187** | 190 | 189 | 189 | 187 |
| furnace_m0_r1_c1 | 1 | 0 | 1 | 1 | **255** | 255 | 255 | 255 | 255 |
| furnace_m1_r0.25_c1 | 1 | 1 | 0.25 | 1 | **255** | 239 | 239 | 255 | 255 |
| furnace_m1_r0.5_c1 | 1 | 1 | 0.5 | 1 | **255** | 221 | 221 | 255 | 255 |
| furnace_m1_r0.75_c1 | 1 | 1 | 0.75 | 1 | **255** | 202 | 202 | 255 | 255 |
| furnace_m1_r0_c1 | 1 | 1 | 0 | 1 | **255** | 255 | 255 | 255 | 255 |
| furnace_m1_r1_c1 | 1 | 1 | 1 | 1 | **255** | 179 | 179 | 255 | 255 |
| half_m0_r0_c0.5 | 0.5 | 0 | 0 | 0.5 | **138** | 143 | 140 | 140 | 139 |
| half_m0_r1_c0.5 | 0.5 | 0 | 1 | 0.5 | **137** | 139 | 138 | 138 | 137 |
| half_m0_r1_c1 | 0.5 | 0 | 1 | 1 | **186** | 189 | 188 | 188 | 186 |
| half_m1_r0.25_c1 | 0.5 | 1 | 0.25 | 1 | **186** | 175 | 175 | 188 | 186 |
| half_m1_r0.5_c1 | 0.5 | 1 | 0.5 | 1 | **186** | 162 | 162 | 188 | 186 |
| half_m1_r0.75_c1 | 0.5 | 1 | 0.75 | 1 | **186** | 147 | 147 | 188 | 186 |
| half_m1_r0_c1 | 0.5 | 1 | 0 | 1 | **186** | 188 | 188 | 188 | 186 |
| half_m1_r1_c1 | 0.5 | 1 | 1 | 1 | **186** | 130 | 130 | 188 | 186 |

| Model | max \|Δ\| | mean \|Δ\| | exact rows (of 26) |
|---|---|---|---|
| single | 76 | 13.92 | 8 |
| split | 76 | 13.38 | 8 |
| **multi** | **2** | **0.96** | 12 |
| **multi γ2.2** | **1** | **0.08** | **24** |

## Findings

1. **X_ITE compensates for multiple scattering (C, from the table).** A white metal stays at 255 for every roughness (186 at intensity 0.5) in X_ITE. Single-scattering split-sum loses up to 30 % of the light at roughness 1 (179 and 130). Fdez-Agüera's compensation restores it exactly. This settles the Loom's open question from finding 4 ("energy compensation, or a lookup with no roughness term?"): it is compensation, or something numerically indistinguishable from it on these 26 rows.
2. **The rest is the encode curve (C).** With X_ITE's own 2.2 encode, 24 of the 26 rows match X_ITE to the byte. The other two are off by 1, and both are the smooth dielectric (roughness 0, base 0.5), where Karis's analytic DFG fit and X_ITE's lookup table differ slightly (L). With the standard sRGB encode, every lit row sits 1–2 bytes above X_ITE, as the Loom's finding 1 predicted.
3. **Dielectrics need energy conservation (C).** Without the split (single), smooth plastic overshoots by up to 6 bytes.
4. **Controls hold (C).** No EnvironmentLight, or intensity 0, gives a black sphere, as in X_ITE. Only PhysicalMaterial is lit by the environment.

## What this does not show

- **Non-uniform environments.** A white furnace cannot tell whether cube faces are oriented right, or whether prefiltering is accurate. Only the integrals are tested. The `09_environment.x3d` sky is a visual check (golden image), not a comparison with X_ITE.
- X_ITE on a real GPU, or any other engine.
- Specular-only versus diffuse-only cubes: the probe used the same cube for both.
- `diffuseCoefficients`: approximated, reported in the loss report.
- `ImageCubeMapTexture` and `GeneratedCubeMapTexture`: unsupported, reported in the loss report.
