# Render receipt schema (format-agnostic core)

`x3d-wgpu-conformance/2` shares these top-level fields with `loom.render-receipt/1` (the glTF Loom). A glTF renderer can emit the same block without any X3D concept.

```json
{
  "schema": "x3d-wgpu-conformance/2",
  "format": "x3d",                       // "x3d" | "gltf" | ... : what was rendered
  "gpu": { "api": "Vulkan", "adapter": "llvmpipe (LLVM 21.1.8, 256 bits)", "backend": "Vulkan" },
  "api_validation": {
    "layer": "VK_LAYER_KHRONOS_validation",
    "layer_version": "1.3.275 (implementation_version 1)",
    "layer_version_source": "<manifest path>",
    "mode": "Require", "requested": true, "enabled": true,
    "errors": 0, "unexpected_errors": 0, "warnings": 0,
    "message_ids": [], "messages": [],
    "known_errors": [{"id": "", "why": ""}],
    "suppressed_by_engine": [{"id": "", "why": ""}],
    "log_sha256": "<hex>",
    "attribution": "..."
  },
  "api_validation_failure": null          // string when the gate failed
}
```

Rules: `enabled:false` under `mode:Require` is a failure. `unexpected_errors` counts error ids not in `known_errors`. Anything format-specific (rows, validators, expected values) lives beside this block, never inside it.

Two keys repeat a fact and are kept for older readers: `receipt` holds the same value as `schema` (the code wrote only `receipt` until 20 Nov 2026, while this page said `schema`), and `adapter` repeats `gpu` with the field names of `x3d-wgpu-conformance/1`. New readers use `schema` and `gpu`.
