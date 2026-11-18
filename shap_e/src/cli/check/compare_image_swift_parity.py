"""
Compare upstream Torch image tensors with Swift image-parity dumps.

Usage:
  python -m cli.check.compare_image_swift_parity

The Python reference is intentionally upstream Torch/OpenAI CLIP rather than
the text-only Python MLX runtime. Thresholds separate preprocessing (where
CoreGraphics and PIL/torchvision interpolate differently) from the neural
components, which receive one shared input and should be near-exact.

Docs: shap_e/parity/README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cli.check.color import GREEN, RED, colored_print
from cli.check.compare_mlx_swift_parity import _metrics


CHECKS = (
    # display, reference file, Swift file, minimum correlation, maximum mean abs
    ("clip_input", "python_clip_input.npy", "swift_clip_input.npy", 0.99, 7.5e-2),
    ("image_grid", "python_image_grid.npy", "swift_image_grid.npy", 0.999, 1e-4),
    (
        "image300m_from_python_grid",
        "python_image300m_output.npy",
        "swift_image300m_output_from_python_grid.npy",
        0.999,
        1e-4,
    ),
    (
        "image300m_full_chain",
        "python_image300m_output.npy",
        "swift_image300m_output.npy",
        0.999,
        1e-4,
    ),
)


def check_mesh(parity: Path, *, required: bool) -> bool:
    path = parity / "swift_image_mesh_summary.json"
    if not path.exists():
        status = "FAIL" if required else "SKIP"
        print(f"{'image_mesh':32s}  {status:6s}  missing {path.name}")
        return not required

    report = json.loads(path.read_text(encoding="utf-8"))
    ok = True
    counts: dict[str, tuple[int, int]] = {}
    for backend in ("cpu", "metal"):
        stats = report.get(backend)
        if not isinstance(stats, dict):
            print(f"{'image_mesh_' + backend:32s}  FAIL    missing backend stats")
            ok = False
            continue
        counts[backend] = (int(stats["vertexCount"]), int(stats["faceCount"]))
        passed = (
            stats["vertexCount"] > 0
            and stats["faceCount"] > 0
            and stats["boundaryEdgeCount"] == 0
            and stats["nonManifoldEdgeCount"] == 0
            and stats["invalidFaceCount"] == 0
            and stats["isClosed"] is True
        )
        ok = ok and passed
        print(
            f"{'image_mesh_' + backend:32s}  {'PASS' if passed else 'FAIL':6s}  "
            f"V={stats['vertexCount']} F={stats['faceCount']} "
            f"boundary={stats['boundaryEdgeCount']} "
            f"nonmanifold={stats['nonManifoldEdgeCount']} "
            f"invalid={stats['invalidFaceCount']}"
        )
    if len(counts) == 2:
        equal = counts["cpu"] == counts["metal"]
        ok = ok and equal
        print(
            f"{'image_mesh_cpu_metal_counts':32s}  {'PASS' if equal else 'FAIL':6s}  "
            f"cpu={counts['cpu']} metal={counts['metal']}"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Torch python_image_* vs Swift swift_image_* dumps."
    )
    parser.add_argument("--parity-dir", default=None, help="Dump directory (default: shap_e/parity).")
    parser.add_argument(
        "--allow-missing-mesh",
        action="store_true",
        help="Accept a component-only run made with ShapECLI --skip-mesh.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    parity = (
        Path(args.parity_dir).expanduser().resolve()
        if args.parity_dir
        else repo_root / "shap_e" / "parity"
    )
    if not parity.is_dir():
        raise SystemExit(f"Parity dir not found: {parity}")

    print(f"Image parity dir: {parity}")
    print(f"{'tensor':32s}  {'status':6s}  {'corr':>8s}  {'mean_abs':>10s}  {'max':>10s}")
    print("-" * 76)
    all_pass = True

    for display, ref_name, swift_name, min_corr, max_mean in CHECKS:
        ref_path = parity / ref_name
        swift_path = parity / swift_name
        if not ref_path.exists() or not swift_path.exists():
            print(f"{display:32s}  FAIL    missing {ref_name} or {swift_name}")
            all_pass = False
            continue
        metrics = _metrics(np.load(ref_path), np.load(swift_path))
        if not metrics.get("ok"):
            print(f"{display:32s}  FAIL    {metrics.get('error')}")
            all_pass = False
            continue
        passed = metrics["corr"] >= min_corr and metrics["mean"] <= max_mean
        all_pass = all_pass and passed
        print(
            f"{display:32s}  {'PASS' if passed else 'FAIL':6s}  "
            f"{metrics['corr']:8.5f}  {metrics['mean']:10.4e}  {metrics['max']:10.4e}"
        )
        if not passed:
            reasons = []
            if metrics["corr"] < min_corr:
                reasons.append(f"corr {metrics['corr']:.5f} < {min_corr}")
            if metrics["mean"] > max_mean:
                reasons.append(f"mean {metrics['mean']:.4e} > {max_mean:g}")
            print(f"{'':32s}          ({'; '.join(reasons)})")

    print("-" * 76)
    all_pass = check_mesh(parity, required=not args.allow_missing_mesh) and all_pass
    print()
    if all_pass:
        colored_print("All image Torch↔Swift checks PASSED.", GREEN)
    else:
        colored_print("Some image Torch↔Swift checks FAILED.", RED)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
