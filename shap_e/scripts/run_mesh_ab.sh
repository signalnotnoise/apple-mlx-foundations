#!/usr/bin/env bash
# Same-SDF CPU vs Metal marching-cubes A/B for Shap-E hole diagnosis.
#
# Usage (repo root):
#   ./shap_e/scripts/run_mesh_ab.sh
#   ./shap_e/scripts/run_mesh_ab.sh --prompt "a red chair" --seed 7 --steps 32 --grid 64
#
# Requires: runtime_bundle/, CLIP BPE, Xcode (for Metal metallib).
# Docs: shap_e/MESH_AB.md
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXTRA_ARGS=("$@")

echo "==> Build ShapECLI (xcodebuild for MLX metallib)"
cd shap_e/swift
xcodebuild -scheme ShapECLI -destination 'platform=macOS' \
  -derivedDataPath ./.derivedData build 2>&1 | tail -25
CLI="${ROOT}/shap_e/swift/.derivedData/Build/Products/Debug/ShapECLI"
cd "$ROOT"

if [[ ! -x "$CLI" ]]; then
  echo "ShapECLI not found at $CLI"
  exit 1
fi

OUT="${ROOT}/shap_e/mesh_ab"
mkdir -p "$OUT"

echo "==> Run mesh A/B"
"$CLI" --ab-mesh --out-dir "$OUT" "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Outputs in $OUT:"
ls -la "$OUT" || true
echo "See shap_e/MESH_AB.md for how to interpret boundary_edges / VERDICT."
