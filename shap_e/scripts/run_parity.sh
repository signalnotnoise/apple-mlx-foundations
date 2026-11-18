#!/usr/bin/env bash
# End-to-end Shap-E Python MLX ↔ Swift parity (L2).
#
# Layers:
#   L0 (optional): SHAP_E_PARITY_PREFLIGHT=1 runs existing cli/check Torch↔MLX scripts
#   L1: dump --check-golden compares summary to committed fixture
#   L2: Swift dump + compare_mlx_swift_parity
#
# Usage (repo root):
#   source .mlx_venv/bin/activate
#   ./shap_e/scripts/run_parity.sh
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

if [[ "${SHAP_E_PARITY_PREFLIGHT:-0}" == "1" ]]; then
  echo "==> 0/3 L0/L1 preflight (existing check scripts)"
  python -m cli.check.check_mlx_runtime_golden
  echo "    (optional deeper L0: check_mlx_karras_loop_parity, check_mlx_decoder_params_parity)"
fi

echo "==> 1/3 Python MLX dump (L1 case + full tensors)"
python -m cli.check.dump_mlx_swift_parity --check-golden --progress

echo "==> 2/3 Build + run Swift dump (xcodebuild for Metal metallib)"
cd "${ROOT}/shap_e/swift"
xcodebuild -scheme ShapECLI -destination 'platform=macOS' \
  -derivedDataPath ./.derivedData build | tail -30
CLI="${ROOT}/shap_e/swift/.derivedData/Build/Products/Debug/ShapECLI"
if [[ ! -x "$CLI" ]]; then
  # Product may land under ShapECLI-macosx or similar depending on Xcode
  CLI="$(find "${ROOT}/shap_e/swift/.derivedData/Build/Products" -type f -name ShapECLI 2>/dev/null | head -1)"
fi
if [[ -z "${CLI}" || ! -x "${CLI}" ]]; then
  echo "ShapECLI binary not found under shap_e/swift/.derivedData"
  exit 1
fi
cd "$ROOT"
"$CLI" --parity \
  --bundle "${ROOT}/runtime_bundle" \
  --parity-dir "${ROOT}/shap_e/parity"

echo "==> 3/3 Compare L2"
python -m cli.check.compare_mlx_swift_parity --parity-dir "${ROOT}/shap_e/parity"
echo "Done. See shap_e/parity/ and shap_e/parity/README.md"
