# Lavapipe conformance crash investigation and patch

The original native crash has **not been reduced to a specific Mesa, LLVM,
wgpu, upload, or application defect**. This patch makes the conformance GPU
fixture exclusive, adds a required cold-cache run under normal libtest
scheduling, and preserves enough context to investigate another native failure.
It is a bounded test-harness mitigation with regression coverage, not proof
that lavapipe generally cannot support concurrent devices.

## Revision and CI evidence

The repository state observed during this investigation was:

| Observation | Source |
| --- | --- |
| Initially, `main` was `ec3f0a4` (workspace version `0.4.0`) and the EnvironmentLight PR was open. | [`main` commit](https://github.com/hagopjay/x3d.wgpu/commit/ec3f0a4937a096fe8a939e01bb4462d825264feb), [PR #4](https://github.com/hagopjay/x3d.wgpu/pull/4); verified with `git ls-remote origin refs/heads/main` and the public pull-request API. |
| EnvironmentLight commit `6861c41` failed the lavapipe workspace-test step in both reported runs. The supplied failure was SIGSEGV in `tests/conformance.rs`. | [Run 44554109959, job 135665392877](https://github.com/hagopjay/x3d.wgpu/actions/runs/44554109959/job/135665392877), [run 44554072144](https://github.com/hagopjay/x3d.wgpu/actions/runs/44554072144); native-signal detail is the issue reporter's log observation. |
| PR follow-up `67a955d` added serial CI tests, `RUST_BACKTRACE=1`, runner diagnostics, and an optional parallel conformance replay. Both subsequent workflow runs succeeded. | [Commit 67a955d](https://github.com/hagopjay/x3d.wgpu/commit/67a955d09a9c9658259c4131cc70f8fb5c07b334), [run 44556120849](https://github.com/hagopjay/x3d.wgpu/actions/runs/44556120849), [run 44556107352](https://github.com/hagopjay/x3d.wgpu/actions/runs/44556107352). |
| The successful replays were not exclusively serial: the optional parallel conformance step also succeeded in both green jobs. | [Jobs API for run 44556120849](https://api.github.com/repos/hagopjay/x3d.wgpu/actions/runs/44556120849/jobs), [jobs for run 44556107352](https://api.github.com/repos/hagopjay/x3d.wgpu/actions/runs/44556107352/jobs), step `Parallel conformance (diagnostic only; tells us if serial is what fixes it)`. |
| The reporter could not reproduce with Mesa `25.2.8` and `4` vCPUs; all `3` conformance tests passed there. | Issue reporter's supplied environment and result; not a measurement made by this patch. |

Because the successful parallel replay followed serial rendering in the same
job, it could have benefited from a warm driver cache. An intermittent race,
cold shader compilation, and driver-specific behavior remain plausible. The
observations do not establish that serialization alone fixed the original
failure. The original downloadable job logs required repository-admin access
through the unauthenticated API, so runner CPU flags and the original native
stack were not available for inspection.

A related upstream report strengthens the reason to capture cold compiler
stacks: [Mesa issue 11797](https://gitlab.freedesktop.org/mesa/mesa/-/issues/11797)
reports intermittent lavapipe crashes in graphical tests on GitHub Actions,
without local reproduction. Its native trace reaches `LLVMGetPointerToGlobal`,
`MCJIT`, and `llvm::PassRegistry::registerPass`, where heap corruption is
reported. The linked [LLVM issue 105547](https://github.com/llvm/llvm-project/issues/105547)
reports that changing the runner and Mesa package reduced but did not eliminate
that separate failure. This is **analogous upstream evidence**, not an
identification of this repository's crash: there is no matching native trace
here. Capture the first cold render's compilation/submission/readback stage and
native frames before concluding that either report is the same defect.

During validation, PR #4 merged as [`fbf36c0`](https://github.com/hagopjay/x3d.wgpu/commit/fbf36c0ee004d6991f792a86018793a35caf9767). The final patch is based directly on that current `main`, whose tree matches the reviewed PR tip `67a955d`. It does not duplicate the EnvironmentLight implementation. The workspace patch version is recorded in [Cargo.toml](../Cargo.toml); a local version change is not a published release.

## What the source audit establishes

| Check | Finding and source |
| --- | --- |
| Test overlap | The original [conformance test binary](../crates/x3d-render/tests/conformance.rs) had `3` tests, each constructing its own `Gpu` and running the complete conformance suite. The encode comparison introduced an additional complete-suite test; [conform.rs](../crates/x3d-render/src/conform.rs) renders each valid row using both sRGB and gamma-`2.2` encoding. Default libtest scheduling can therefore overlap device creation, pipeline compilation, rendering, and teardown. |
| Cube dimensions and mips | [`environment::SIZE`, `mip_levels`, and `prefilter`](../crates/x3d-render/src/environment.rs) define a face side of `64`, `7` levels derived as `ilog2(SIZE) + 1`, and level sides `64, 32, 16, 8, 4, 2, 1`. Every level has `6` heap-backed face vectors with `side × side` RGBA half-float texels. These are source-derived dimensions, not a measurement of driver acceptance. |
| Upload layout | [`upload_env_cube`](../crates/x3d-render/src/renderer.rs) uses `Rgba16Float`, a `D2` texture with `6` layers, and a cube view. Each write addresses one face and mip with `8 × side` bytes per row and `side` rows. The black fallback uses a face side of `1` and one mip. The copy-buffer `256`-byte row-alignment rule does not apply to `Queue::write_texture`; see the [wgpu layout contract](https://docs.rs/wgpu/30.0.2/wgpu/struct.TexelCopyBufferLayout.html#structfield.bytes_per_row), with the dependency version recorded in [Cargo.lock](../Cargo.lock). |
| CPU prefilter stack | [`prepare`, `project_sh9`, `prefilter`, and `prefiltered`](../crates/x3d-render/src/environment.rs) are nonrecursive; image texels and mip output are in `Vec`s. Their source does not show a large stack-resident image array. This does not exclude a native LLVM/compiler stack failure. |
| A crash before the first environment row | [`Renderer::new` and `set_environment`](../crates/x3d-render/src/renderer.rs) initialize/bind a black cube even for scenes without EnvironmentLight. The common scene shader includes the environment branch. Therefore an early failure does not by itself isolate the CPU prefilter or an environment-lit scene. |
| Rust backtraces | `RUST_BACKTRACE=1` helps with Rust panics. A native SIGSEGV does not unwind Rust and needs native stack information. Rust's documented spawned-thread default is currently `2 MiB` on Tier-1 platforms, but this patch does not increase it without evidence of stack exhaustion; see [std::thread stack-size documentation](https://doc.rust-lang.org/std/thread/index.html#stack-size). |

No tolerance, expected image, conformance measurement, environment resolution,
float format, CPU feature override, or LLVM worker limit is changed to make the
suite pass.

## Mitigation and regression guard

The [conformance fixture](../crates/x3d-render/tests/conformance.rs) holds a
process-local mutex from before `Gpu::headless` until after all fixture-owned GPU
handles drop. The GPU field is declared before the guard field so Rust's
field-drop order releases the device before permitting another test to enter.
It protects a normal local `cargo test` as well as CI. It does not serialize
independent processes or change production renderer concurrency.

The [lavapipe workflow](../.github/workflows/ci.yml) now:

- Selects lavapipe's ICD explicitly and records the exact revision, Rust build,
  Mesa/LLVM packages, CPU flags/affinity, stack setting, Vulkan adapter, and
  queried renderer budget.
- Runs conformance with normal libtest scheduling as a **required** step before
  the serial workspace run. That step uses a new `MESA_SHADER_CACHE_DIR` and
  `X3D_PIPELINE_CACHE=off` to avoid relying on a preceding render's caches. The
  fixture still prevents simultaneous conformance devices; this checks the
  mitigation under parallel scheduling, not raw driver concurrency.
- Runs `cargo test --workspace -- --test-threads=1 --nocapture`, retains
  `RUST_BACKTRACE=1`, and saves uncaptured output with shell `pipefail` so `tee`
  cannot conceal a failing test process.
- Emits test/adapter and row/scene/source/encode context before entering the
  driver. [Headless render stages](../crates/x3d-render/src/headless.rs) provide
  finer context when its debug logging is enabled.
- Fails conformance on uncaptured wgpu validation errors. Previously the
  application's logging callback could let a test continue after invalid GPU
  work.
- Attempts a bounded native GDB replay after a test-step failure and uploads
  its output together with the failed test logs. A replay may pass if the
  original failure was intermittent; it does not replace or clear that failure.

If the crash recurs, use the last emitted row/encode/stage and native stack to
choose a discriminating experiment. LLVM compilation frames suggest a compiler
or compiler-stack investigation; a fault inside JIT-generated sampling code
suggests testing the identified face/mip/format and CPU-vector path. A native
stack and process mappings are needed before labelling either as the cause.

## Local validation

The accompanying evidence archive contains the command, environment, output, and return-code records named below. These are correctness checks, not comparable performance benchmarks. The patched cold run overlapped a separate headless compilation; its duration should not be compared with the controlled baseline timings. No GPU adapter was skipped: `X3D_REQUIRE_GPU=1` was set for the GPU test runs.

| Validation | Result | Evidence |
| --- | --- | --- |
| Original `6861c41`, cold-cache default scheduling | `3` passed; `0` failed | `evidence/baseline-parallel.log`, `baseline-parallel.command.json`, `baseline-parallel.result.json`. |
| Original `6861c41`, cold-cache serial scheduling | `3` passed; `0` failed | `evidence/baseline-serial.log`, `baseline-serial.command.json`, `baseline-serial.result.json`. |
| Patched conformance, cold-cache default scheduling | `3` passed; `0` failed | `evidence/patched-parallel.log`, `patched-parallel.command.json`, `patched-parallel.result.json`. |
| Serial workspace tests, including doctests | `117` passed; `0` failed; `0` ignored | Sum of the libtest `test result` lines in `evidence/workspace-serial.log`; command and exit status in `workspace-serial.command.json` and `workspace-serial.result.json`. |
| Formatting and Clippy | Passed | Commands and exit statuses in `evidence/checks.json`, output in `fmt.log` and `clippy.log`. |
| Headless-only build | Passed | `cargo build --locked -p x3d-render --no-default-features`; `evidence/headless-build.log` and `checks.json`. |
| Workflow YAML and shell parsing | Passed | `evidence/workflow-checks.json`; native GDB execution on GitHub remains unverified. |

Local adapter provenance: Mesa `25.0.7-2+deb13u1`, LLVM `19.1.7` and software Vulkan, as reported by `vulkaninfo` and recorded with CPU flags, affinity, package checksum and Rust build in `evidence/environment.json`. Neither baseline scheduling mode reproduced the reported signal. Local GDB launch was blocked by `ptrace` permissions, so no native backtrace is claimed.

## Replaying the original failure on the runner

The separate manual [native-crash workflow](../.github/workflows/lavapipe-diagnose.yml) checks out the exact reported failing commit, rather than the guarded fixture. It executes cold-cache default-parallel and serial cases under GDB on the same runner and preserves native stacks, mappings, registers, fault instruction, per-case exit status, and environment. The second case still runs if the first fails. This allows the original trigger to be studied after required CI has adopted the mitigation. It has been syntax-checked and its status handling exercised with simulated outcomes; it has not been executed on GitHub from this session.
