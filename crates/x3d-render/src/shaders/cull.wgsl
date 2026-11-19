// GPU frustum culling: one thread per draw. A draw whose bounding sphere is
// fully outside any of the six frustum planes gets instance_count = 0 in its
// indirect-draw arguments, so the GPU skips it without the CPU deciding.

struct Cull {
    planes: array<vec4f, 6>, // ax + by + cz + d ≥ 0 inside, normalised
    count: vec4u,            // x = number of draws
}

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> spheres: array<vec4f>;         // xyz centre, w radius (world)
@group(0) @binding(2) var<storage, read_write> args: array<u32>;         // 5 u32 per draw (DrawIndexedIndirectArgs)
@group(0) @binding(3) var<storage, read_write> visible: atomic<u32>;

@compute @workgroup_size(64)
fn cull_main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= cull.count.x { return; }
    let s = spheres[i];
    var inside = 1u;
    for (var p = 0u; p < 6u; p++) {
        let plane = cull.planes[p];
        if dot(plane.xyz, s.xyz) + plane.w < -s.w { inside = 0u; }
    }
    args[i * 5u + 1u] = inside; // instance_count
    if inside == 1u { atomicAdd(&visible, 1u); }
}
