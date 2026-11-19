#!/usr/bin/env bash
# Build x3d.wgpu for the browser: web/dist/ is a static site (serve it with any
# HTTP server, e.g. `python3 -m http.server -d web/dist`).
# Needs: rustup target add wasm32-unknown-unknown; cargo install wasm-bindgen-cli --version 0.2.130
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build -p x3d-web --target wasm32-unknown-unknown --profile web
rm -rf web/dist && mkdir -p web/dist/scenes
wasm-bindgen --target web --no-typescript --out-dir web/dist/pkg target/wasm32-unknown-unknown/web/x3d_web.wasm
cp web/index.html web/dist/
for f in 01_primitives 02_indexed_face_sets 03_def_use_switch 04_lines_and_points 05_still_life 06_terrain 07_strips_and_fans; do
  cp scenes/$f.x3d web/dist/scenes/
done
ls -l web/dist/pkg
