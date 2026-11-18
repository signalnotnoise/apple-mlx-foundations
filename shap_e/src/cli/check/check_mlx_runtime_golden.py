from __future__ import annotations

import argparse
from pathlib import Path

from mlx_shap_e.runtime_core.golden import (
    RuntimeGoldenCase,
    compare_runtime_summary,
    load_golden,
    save_golden,
)
from cli.check.color import colored_print, GREEN, RED


def main() -> None:
    
    parser = argparse.ArgumentParser(
        description="Run the MLX runtime against a committed golden fixture."
    )
    parser.add_argument(
        "--golden-json",
        default=None,
        help="Path to the runtime golden JSON fixture.",
    )
    parser.add_argument(
        "--runtime-artifacts-dir",
        default=None,
        help="Runtime bundle directory containing manifest.json plus tensor blobs.",
    )
    parser.add_argument(
        "--clip-bpe-path",
        default=None,
        help="Optional path to CLIP tokenizer merges file.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Overwrite the golden fixture with the current runtime output.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    golden_path = (
        Path(args.golden_json).expanduser().resolve()
        if args.golden_json
        else repo_root / "tests" / "fixtures" / "runtime_golden_a_red_chair_steps2.json"
    )
    runtime_artifacts_dir = (
        Path(args.runtime_artifacts_dir).expanduser().resolve()
        if args.runtime_artifacts_dir
        else repo_root / "runtime_bundle"
    )

    expected = load_golden(golden_path)
    case = RuntimeGoldenCase(**expected["case"])
    from mlx_shap_e.runtime_core.pipeline import MLXShapERuntime
    from mlx_shap_e.runtime_core.parity_tensors import run_mlx_parity_case

    # Shared pipeline with dump_mlx_swift_parity (L1 summary + L2 tensors stay in sync).
    # Vertex colors are included so the committed golden mesh heads stay comparable.
    runtime = MLXShapERuntime(runtime_artifacts_dir, tokenizer_bpe_path=args.clip_bpe_path)
    payload = run_mlx_parity_case(
        runtime,
        case,
        clip_bpe_path=args.clip_bpe_path,
        progress=True,
        include_vertex_colors=True,
    )
    actual = payload["summary"]

    if args.update:
        save_golden(golden_path, actual)
        print(f"Updated runtime golden fixture: {golden_path}")
        return

    try:
        compare_runtime_summary(actual, expected, atol=1e-4)
        colored_print(f"\nRuntime golden check passed: {golden_path}", GREEN)
    except AssertionError as e:
        colored_print(f"\nRuntime golden check failed: {e}", RED)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
