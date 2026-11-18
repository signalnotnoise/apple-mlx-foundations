#!/usr/bin/env bash
# End-to-end image-conditioning validation: upstream Torch oracle, Swift
# preprocessing/components, and CPU/Metal mesh-topology smoke.
#
# Usage (repo root):
#   ./shap_e/scripts/run_image_parity.sh
#
# Optional environment:
#   SHAP_E_IMAGE_PARITY_DIR  Dump directory (default: ${TMPDIR:-/tmp}/shap-e-image-parity)
#   SHAP_E_IMAGE_BUNDLE      Runtime bundle (default: runtime_bundle)
#   SHAP_E_IMAGE_CACHE       Checkpoint cache (default: shap_e_model_cache)
#   SHAP_E_IMAGE_PATH        Optional input image; default synthetic case
#   SHAP_E_IMAGE_STEPS       Mesh-smoke steps (default: 64)
#   SHAP_E_IMAGE_GRID        Mesh-smoke grid (default: 64)
#   SHAP_E_IMAGE_GUIDANCE    Mesh-smoke guidance (default: 3)
#   SHAP_E_SKIP_MESH=1       Stop after neural component tensors
#
# Docs: shap_e/parity/README.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

VENV="${ROOT}/.mlx_venv/bin/activate"
if [[ -f "$VENV" ]]; then
  # shellcheck disable=SC1090
  source "$VENV"
else
  echo "Note: .mlx_venv not found; using current Python."
fi

export PYTHONPATH="${ROOT}/shap_e/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${SHAP_E_IMAGE_PARITY_DIR:-${TMPDIR:-/tmp}/shap-e-image-parity}"
BUNDLE="${SHAP_E_IMAGE_BUNDLE:-${ROOT}/runtime_bundle}"
CACHE="${SHAP_E_IMAGE_CACHE:-${ROOT}/shap_e_model_cache}"
STEPS="${SHAP_E_IMAGE_STEPS:-64}"
GRID="${SHAP_E_IMAGE_GRID:-64}"
GUIDANCE="${SHAP_E_IMAGE_GUIDANCE:-3}"
mkdir -p "$OUT"

echo "==> 1/3 Upstream Torch image dump"
IMAGE_ARGS=()
if [[ -n "${SHAP_E_IMAGE_PATH:-}" ]]; then
  IMAGE_ARGS=(--image-path "$SHAP_E_IMAGE_PATH")
fi
if [[ ${#IMAGE_ARGS[@]} -gt 0 ]]; then
  python -m cli.check.dump_image_swift_parity \
    --out-dir "$OUT" \
    --cache-dir "$CACHE" \
    "${IMAGE_ARGS[@]}"
else
  python -m cli.check.dump_image_swift_parity \
    --out-dir "$OUT" \
    --cache-dir "$CACHE"
fi

echo "==> 2/3 Build + run Swift image dump"
cd "${ROOT}/shap_e/swift"
xcodebuild -scheme ShapECLI -destination 'platform=macOS' \
  -derivedDataPath ./.derivedData build | tail -30
CLI="${ROOT}/shap_e/swift/.derivedData/Build/Products/Debug/ShapECLI"
if [[ ! -x "$CLI" ]]; then
  CLI="$(find "${ROOT}/shap_e/swift/.derivedData/Build/Products" -type f -name ShapECLI 2>/dev/null | head -1)"
fi
if [[ -z "${CLI}" || ! -x "${CLI}" ]]; then
  echo "ShapECLI binary not found under shap_e/swift/.derivedData"
  exit 1
fi
cd "$ROOT"
SWIFT_ARGS=(
  --image-parity
  --bundle "$BUNDLE"
  --parity-dir "$OUT"
  --steps "$STEPS"
  --guidance "$GUIDANCE"
  --grid "$GRID"
  --query-batch 65536
)
COMPARE_ARGS=(--parity-dir "$OUT")
if [[ "${SHAP_E_SKIP_MESH:-0}" == "1" ]]; then
  SWIFT_ARGS+=(--skip-mesh)
  COMPARE_ARGS+=(--allow-missing-mesh)
fi
"$CLI" "${SWIFT_ARGS[@]}"

echo "==> 3/3 Compare image tensors and mesh topology"
python -m cli.check.compare_image_swift_parity "${COMPARE_ARGS[@]}"
echo "Done. Image parity outputs are in $OUT"
