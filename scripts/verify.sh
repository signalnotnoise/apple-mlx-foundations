#!/usr/bin/env bash
# Unified local verification entry point.
#
# Usage:
#   ./scripts/verify.sh [--quick] [--base REF]
#
# --quick runs standards, graph freshness, Ruff, focused Python tests, and
# swift-format diagnostics for changed Swift files. Full mode additionally
# runs the root Python suite, the Swift test suites, and RuntimeWrapper
# build and tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUICK=0
BASE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --base)
      BASE="${2:-}"
      if [[ -z "$BASE" ]]; then
        echo "--base requires a Git revision" >&2
        exit 2
      fi
      shift 2
      ;;
    -h|--help)
      sed -n '1,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -x "${ROOT}/.mlx_venv/bin/python" ]]; then
  PY="${PYTHON:-${ROOT}/.mlx_venv/bin/python}"
else
  PY="${PYTHON:-python3}"
fi

echo "==> Coding standards"
if [[ -n "$BASE" ]]; then
  python3 scripts/check_standards.py --base "$BASE"
else
  python3 scripts/check_standards.py --changed
fi

echo "==> Knowledge graph"
python3 knowledge_graph/update_graph.py --check

echo "==> Ruff (critical correctness rules)"
"$PY" -m ruff check shap_e/src tests scripts

echo "==> Focused Python tests"
PYTHONPATH="${ROOT}/shap_e/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m pytest tests/test_check_standards.py tests/test_download_shap_e_bundle.py -q

echo "==> swift-format (changed Swift files, diagnostics only)"
SWIFT_FORMAT=""
if command -v swift-format >/dev/null 2>&1; then
  SWIFT_FORMAT="$(command -v swift-format)"
elif command -v xcrun >/dev/null 2>&1; then
  SWIFT_FORMAT="$(xcrun --find swift-format 2>/dev/null || true)"
fi
if [[ -n "$SWIFT_FORMAT" ]]; then
  if [[ -n "$BASE" ]]; then
    CHANGED_SWIFT="$(git diff --name-only --diff-filter=ACMR "$BASE...HEAD" -- '*.swift' || true)"
  else
    CHANGED_SWIFT="$({ git diff --name-only --diff-filter=ACMR HEAD -- '*.swift'; git ls-files --others --exclude-standard -- '*.swift'; } | sort -u)"
  fi
  if [[ -n "$CHANGED_SWIFT" ]]; then
    # Existing files were not mass-formatted when this gate was introduced;
    # diagnostics are visible but do not override the standards checker's
    # anti-ratchet errors. New code should still be formatted before review.
    while IFS= read -r file; do
      [[ -f "$file" ]] && "$SWIFT_FORMAT" lint "$file" || true
    done <<< "$CHANGED_SWIFT"
  else
    echo "No changed Swift files."
  fi
else
  echo "swift-format not found; skipping style diagnostics."
fi

if [[ "$QUICK" == "1" ]]; then
  echo "Quick verification complete."
  exit 0
fi

echo "==> Root Python tests"
if ! "$PY" -c 'import mlx.core' >/dev/null 2>&1; then
  echo "MLX is not importable in $PY. Create .mlx_venv and install the project first." >&2
  exit 1
fi
PYTHONPATH="${ROOT}/shap_e/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m pytest tests -q

echo "==> ShapERuntime tests"
(cd shap_e/swift && ./run_tests.sh)

echo "==> RuntimeWrapper build"
(cd runtime_wrapper && swift build)

echo "==> RuntimeWrapper tests"
# The real-weights TripoSR mesh test skips unless TRIPOSR_WEIGHTS_PATH and
# TRIPOSR_IMAGE_PATH are set; set them to count it as evidence.
(cd runtime_wrapper && swift test)

echo "==> Apple platform builds (iOS and visionOS)"
DERIVED_ROOT="${TMPDIR:-/tmp}/apple-mlx-foundations-derived"
mkdir -p "$DERIVED_ROOT"
(cd shap_e/swift && xcodebuild build \
  -scheme ShapERuntime-Package \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$DERIVED_ROOT/shap-e-ios" >/dev/null)
(cd shap_e/swift && xcodebuild build \
  -scheme ShapERuntime-Package \
  -destination 'generic/platform=visionOS' \
  -derivedDataPath "$DERIVED_ROOT/shap-e-visionos" >/dev/null)
(cd runtime_wrapper && xcodebuild build \
  -scheme RuntimeWrapper \
  -destination 'generic/platform=iOS' \
  -derivedDataPath "$DERIVED_ROOT/wrapper-ios" >/dev/null)
(cd runtime_wrapper && xcodebuild build \
  -scheme RuntimeWrapper \
  -destination 'generic/platform=visionOS' \
  -derivedDataPath "$DERIVED_ROOT/wrapper-visionos" >/dev/null)
echo "Apple platform builds succeeded (macOS tests; iOS/visionOS compile checks)."

echo "Full verification complete."
