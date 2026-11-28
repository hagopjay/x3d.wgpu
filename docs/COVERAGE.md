# Per-node coverage for the multi-renderer audit

`x3d-view coverage nodes.json [--scenes DIR] [--png-dir DIR] [--json OUT] [--backend vulkan]`

`nodes.json` is an object keyed by node name (the audit's 260-node list works as is). Output is `x3d-wgpu-coverage/1`:

- `static`: what the loader's own table says (`drawn`, `approximated`, `silent`, `skipped`); see `crates/x3d-scene/src/support.rs`. This is a claim from reading the source (Likely).
- With `--scenes`, for each node it loads `DIR/<Node>.x3d` and records `observed` (`loaded`, `approximated`, `ignored`, `skipped`, `load_error`, `render_error`, `no_scene`), the loss report, a PNG, `pixels_differing_from_corner` (a crude "something was drawn" probe, not a quality score), and that node's Vulkan validation block. `loaded` means it loaded and rendered without a loss-report entry; it does not say the picture is right.
- The `static` and `observed` columns should agree; where they don't, the table or the loader is wrong.
- Receipt-level `api_validation` follows docs/RECEIPT-SCHEMA.md. Set `X3D_VVL=require` to fail on unexpected errors.
