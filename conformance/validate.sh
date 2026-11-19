#!/usr/bin/env bash
# Re-run the validators behind conformance/ and refresh validators/MANIFEST.json.
#
#   LOOM=~/src/x3d.loom LOOM_FORK=~/src/x3d_mcp conformance/validate.sh
#
# Runs the Loom's grammar + meaning witnesses (x3d_mcp's validate_xml and
# semantic_diagnostics, read-only, in the fork's own .venv) over every .x3d in
# this repo, copies their evidence here, and records each file's sha256, so
# `cargo test --test conformance` fails if a scene changes without being
# re-validated.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOOM="${LOOM:?set LOOM to an x3d.loom checkout}"
: "${LOOM_FORK:?set LOOM_FORK to an x3d_mcp checkout with a .venv}"
RUN="$(mktemp -d)"
"$LOOM/kit/loom" grammar "$HERE" --evidence "$RUN/evidence" --files 'scenes/*.x3d' >/dev/null || true
cp "$RUN/evidence/grammar.json" "$HERE/conformance/validators/grammar-samples.json"
cp "$RUN/evidence/meaning.json" "$HERE/conformance/validators/meaning-samples.json"
"$LOOM/kit/loom" grammar "$HERE" --evidence "$RUN/evidence2" --files 'conformance/scenes/*.x3d' >/dev/null || true
cp "$RUN/evidence2/grammar.json" "$HERE/conformance/validators/grammar-conformance.json"
cp "$RUN/evidence2/meaning.json" "$HERE/conformance/validators/meaning-conformance.json"
python3 - "$HERE" "$LOOM" <<'PY'
import hashlib, json, subprocess, sys, pathlib
here, loom = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
v = here / "conformance/validators"
files = {}
for kind in ("samples", "conformance"):
    g = json.loads((v / f"grammar-{kind}.json").read_text())
    m = json.loads((v / f"meaning-{kind}.json").read_text())
    for f, r in g["measurements"].items():
        files[f] = {
            "sha256": hashlib.sha256((here / f).read_bytes()).hexdigest(),
            "xsd_valid": r["xsd_valid"], "xsd_errors": r["xsd_errors"], "xsd_version": r["xsd_version"],
            "semantic_errors": m["measurements"][f]["semantic_errors"],
            "semantic_warnings": m["measurements"][f]["semantic_warnings"],
        }
    instrument = g["instrument"]
git = lambda *a: subprocess.run(["git", "-C", str(loom), *a], capture_output=True, text=True).stdout.strip()
manifest = {
    "validator": {k: instrument[k] for k in ("tool", "commit", "branch", "dirty")},
    "witness_kit": {"tool": "x3d.loom kit/loom grammar (grammar + meaning witnesses)", "commit": git("rev-parse", "HEAD")},
    "evidence": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(v.glob("*-*.json"))},
    "files": dict(sorted(files.items())),
}
(v / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"{len(files)} files validated by x3d_mcp@{instrument['commit'][:7]}")
PY
