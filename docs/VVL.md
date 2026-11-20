# Vulkan Validation Layer (VVL) in x3d.wgpu

Every conformance receipt records what `VK_LAYER_KHRONOS_validation` said while the rows ran.

## Switch
`X3D_VVL=on` asks for the layer. `X3D_VVL=require` also **fails** the run when the layer is not actually loaded, or when any error not listed under "Known errors" appears. Unset, wgpu's default applies (layer on in debug builds only). CI sets `require` on the lavapipe job and on the self-hosted GPU job. The serial and cold-cache lavapipe mitigation is unchanged.

## What the receipt records
`api_validation` (receipt level, schema `x3d-wgpu-conformance/2`; per-row blocks carry `errors`, `unexpected_errors`, `warnings`, `message_ids`): layer name, layer version and the manifest it was read from, mode, `enabled`, counts, message ids with counts and first text, `known_errors`, `suppressed_by_engine`, and the sha256 of the captured log. Field meanings are in `docs/RECEIPT-SCHEMA.md`.

## Known errors (allow-list, not clean)
| id | where | status |
|---|---|---|
| SYNC-HAZARD-READ-AFTER-WRITE | wide-line render tests (`lines_have_their_x3d_width`, `a_line_running_behind_the_camera`) | Real finding. A vertex-attribute read follows a staging copy with no barrier. Cause unproven. Tried COPY_DST and an empty submit; neither helped. |

The allow-list means "any error fails" is true only for errors outside this table. The conformance rows themselves are clean (30/30, 0 errors, 0 warnings). Warnings (for example Shader-OutputNotConsumed in ComposedShader tests) are counted, not gated.

## Limits
- wgpu-hal itself silences five VUIDs before we see them; they are listed in every receipt (`suppressed_by_engine`).
- Messages are attributed to rows by time, so run serially.
- Verified on lavapipe, Mesa 25.2.8, layer 1.3.275. A real GPU and a newer layer will say different things; the receipt names which.
- SSIM/PSNR are not in receipts today: rows are probes (pixel values against expected), and goldens use their TOLERANCE file.
