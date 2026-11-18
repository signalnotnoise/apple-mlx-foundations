"""
Dump full MLX pipeline tensors for Swift L2 parity.

Uses the same golden case + MLXShapERuntime path as check_mlx_runtime_golden.
Existing L0/L1 scripts are unchanged; this only adds full-tensor dumps.

Usage (from repo root, with package importable):
  cd shap_e/src
  python -m cli.check.dump_mlx_swift_parity

Docs: shap_e/parity/README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cli.check.color import GREEN, RED, colored_print
from mlx_shap_e.runtime_core.golden import RuntimeGoldenCase, compare_runtime_summary, load_golden
from mlx_shap_e.runtime_core.parity_tensors import run_mlx_parity_case, write_parity_dumps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump MLX runtime tensors for Swift parity (extends golden case)."
    )
    parser.add_argument(
        "--golden-json",
        default=None,
        help="Path to runtime golden JSON fixture (default: tests/fixtures/runtime_golden_a_red_chair_steps2.json).",
    )
    parser.add_argument(
        "--runtime-artifacts-dir",
        default=None,
        help="Runtime bundle directory (default: repo runtime_bundle/).",
    )
    parser.add_argument(
        "--clip-bpe-path",
        default=None,
        help="Optional path to CLIP BPE merges file.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for python_*.npy (default: shap_e/parity).",
    )
    parser.add_argument(
        "--check-golden",
        action="store_true",
        help="Also compare runtime summary against the committed golden (L1).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show sampling progress.",
    )
    args = parser.parse_args()

    # cli/check -> parents[0]=check, [1]=cli, [2]=src, [3]=shap_e, [4]=repo
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
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else repo_root / "shap_e" / "parity"
    )

    expected = load_golden(golden_path)
    case = RuntimeGoldenCase(**expected["case"])
    colored_print(
        f"Dumping MLX parity for case prompt={case.prompt!r} steps={case.steps} "
        f"seed={case.seed} grid={case.mesh_grid_size}",
        GREEN,
    )
    colored_print(f"Bundle: {runtime_artifacts_dir}", GREEN)
    colored_print(f"Out:    {out_dir}", GREEN)

    from mlx_shap_e.runtime_core.pipeline import MLXShapERuntime

    runtime = MLXShapERuntime(runtime_artifacts_dir, tokenizer_bpe_path=args.clip_bpe_path)
    payload = run_mlx_parity_case(
        runtime,
        case,
        clip_bpe_path=args.clip_bpe_path,
        progress=args.progress,
        include_vertex_colors=False,
    )
    write_parity_dumps(out_dir, payload, prefix="python")

    colored_print(
        f"Wrote python_*.npy + python_golden.npz + case.json to {out_dir}",
        GREEN,
    )
    colored_print(
        f"  latent {payload['latent'].shape}, sdf {payload['sdf_field'].shape}, "
        f"verts {payload['verts'].shape}, faces {payload['faces'].shape}, "
        f"params {len(payload['params'])}",
        GREEN,
    )

    if args.check_golden:
        try:
            compare_runtime_summary(payload["summary"], expected, atol=1e-4)
            colored_print(f"L1 golden summary check passed: {golden_path}", GREEN)
        except AssertionError as exc:
            colored_print(f"L1 golden summary check failed: {exc}", RED)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
