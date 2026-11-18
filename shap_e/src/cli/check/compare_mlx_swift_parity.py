"""
Compare Python MLX parity dumps vs Swift dumps (L2 assert-grade).

Usage:
  python -m cli.check.compare_mlx_swift_parity

Expects shap_e/parity/python_*.npy (or python_golden.npz) and swift_*.npy.
Reuses color bands from color.py for display; exit 1 on failed thresholds.

Docs: shap_e/parity/README.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from cli.check.color import GREEN, RED, colored_print


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict:
    if ref.shape != got.shape:
        return {"ok": False, "error": f"shape mismatch ref={ref.shape} got={got.shape}"}
    r = ref.astype(np.float64).reshape(-1)
    g = got.astype(np.float64).reshape(-1)
    diff = np.abs(r - g)
    if r.std() < 1e-12 and g.std() < 1e-12:
        corr = 1.0 if float(diff.max()) < 1e-6 else 0.0
    else:
        corr = float(np.corrcoef(r, g)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    return {
        "ok": True,
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "corr": corr,
    }


def _load_pair(parity: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    py_path = parity / f"python_{name}.npy"
    sw_path = parity / f"swift_{name}.npy"
    if not py_path.exists() or not sw_path.exists():
        return None
    return np.load(py_path), np.load(sw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MLX python_* vs swift_* parity dumps.")
    parser.add_argument(
        "--parity-dir",
        default=None,
        help="Directory with dumps (default: shap_e/parity).",
    )
    parser.add_argument(
        "--metal-mc",
        action="store_true",
        help=(
            "Swift dump used Metal marching cubes. Vertex/face order may differ, "
            "so mesh gates stay count + bbox only."
        ),
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

    # (name, npy stem without prefix, min_corr, max_mean_abs, exact?)
    # exact=True: integer or float allclose with tight atol
    checks: list[tuple[str, str, float | None, float | None, Callable[[np.ndarray, np.ndarray], bool] | None]] = [
        ("token_ids", "token_ids", None, None, lambda a, b: np.array_equal(a, b)),
        ("clip_embedding", "clip_embedding", 0.999, 1e-3, None),
        ("sigmas", "sigmas", None, None, lambda a, b: np.allclose(a, b, atol=1e-5, rtol=0.0)),
        ("latent", "latent", 0.999, 1e-3, None),
        ("pred_xstart", "pred_xstart", 0.999, 1e-3, None),
        ("sdf_field", "sdf_field", 0.999, 1e-3, None),
    ]

    # Dynamic param keys from python dumps
    for path in sorted(parity.glob("python_params_*.npy")):
        stem = path.name[len("python_") : -len(".npy")]  # params_nerstf_...
        checks.append((stem, stem, 0.999, 1e-3, None))

    print(f"Parity dir: {parity}")
    print(f"{'tensor':28s}  {'status':6s}  {'corr':>8s}  {'mean_abs':>10s}  {'max':>10s}")
    print("-" * 72)

    all_pass = True

    for display, stem, min_corr, max_mean, exact_fn in checks:
        pair = _load_pair(parity, stem)
        if pair is None:
            print(f"{display:28s}  FAIL    missing python_{stem}.npy or swift_{stem}.npy")
            all_pass = False
            continue

        ref, got = pair
        if exact_fn is not None:
            if ref.shape != got.shape:
                print(f"{display:28s}  FAIL    shape ref={ref.shape} got={got.shape}")
                all_pass = False
                continue
            ok = bool(exact_fn(ref, got))
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            m = _metrics(ref.astype(np.float64), got.astype(np.float64)) if ref.size else {"corr": 1.0, "mean": 0.0, "max": 0.0, "ok": True}
            print(
                f"{display:28s}  {status:6s}  {m.get('corr', float('nan')):8.5f}  "
                f"{m.get('mean', float('nan')):10.4e}  {m.get('max', float('nan')):10.4e}"
            )
            continue

        m = _metrics(ref, got)
        if not m.get("ok"):
            print(f"{display:28s}  FAIL    {m.get('error')}")
            all_pass = False
            continue

        pass_corr = m["corr"] >= (min_corr or 0.0)
        pass_mean = m["mean"] <= (max_mean if max_mean is not None else float("inf"))
        status = "PASS" if (pass_corr and pass_mean) else "FAIL"
        if status == "FAIL":
            all_pass = False
        color = GREEN if status == "PASS" else RED
        print(
            f"{display:28s}  {status:6s}  {m['corr']:8.5f}  {m['mean']:10.4e}  {m['max']:10.4e}"
        )
        if status == "FAIL":
            reasons = []
            if not pass_corr:
                reasons.append(f"corr {m['corr']:.5f} < {min_corr}")
            if not pass_mean:
                reasons.append(f"mean {m['mean']:.4e} > {max_mean}")
            print(f"{'':28s}          ({'; '.join(reasons)})")

    # Structural mesh checks. CPU MC shares unique-edge order with Python.
    # --metal-mc may permute vertices, so those dumps keep count + bbox only.
    print("-" * 72)
    verts_pair = _load_pair(parity, "verts")
    faces_pair = _load_pair(parity, "faces")
    if verts_pair is None or faces_pair is None:
        print(f"{'mesh':28s}  SKIP    missing verts/faces dumps")
        all_pass = False
    else:
        ref_v, got_v = verts_pair
        ref_f, got_f = faces_pair
        verts_count_ok = ref_v.shape[0] == got_v.shape[0]
        faces_count_ok = ref_f.shape[0] == got_f.shape[0]
        count_ok = verts_count_ok and faces_count_ok
        if not verts_count_ok:
            all_pass = False
        if not faces_count_ok:
            all_pass = False
        print(
            f"{'mesh_vert_count':28s}  {'PASS' if verts_count_ok else 'FAIL':6s}  "
            f"ref={ref_v.shape[0]} got={got_v.shape[0]}"
        )
        print(
            f"{'mesh_face_count':28s}  {'PASS' if faces_count_ok else 'FAIL':6s}  "
            f"ref={ref_f.shape[0]} got={got_f.shape[0]}"
        )

        if count_ok and ref_v.size and got_v.size:
            bbox_ref = np.stack([ref_v.min(0), ref_v.max(0)])
            bbox_got = np.stack([got_v.min(0), got_v.max(0)])
            bbox_ok = np.allclose(bbox_ref, bbox_got, atol=1e-2, rtol=0.0)
            bbox_status = "PASS" if bbox_ok else "FAIL"
            if not bbox_ok:
                all_pass = False
            print(f"{'mesh_bbox':28s}  {bbox_status:6s}  atol=1e-2")

            m = _metrics(ref_v, got_v)
            if args.metal_mc:
                if m.get("ok"):
                    print(
                        f"{'mesh_verts_corr':28s}  info    {m['corr']:8.5f}  "
                        f"{m['mean']:10.4e}  {m['max']:10.4e}  (order-sensitive)"
                    )
            else:
                if not m.get("ok"):
                    print(f"{'mesh_verts':28s}  FAIL    {m.get('error')}")
                    all_pass = False
                else:
                    pass_corr = m["corr"] >= 0.999
                    pass_mean = m["mean"] <= 1e-3
                    status = "PASS" if (pass_corr and pass_mean) else "FAIL"
                    if status == "FAIL":
                        all_pass = False
                    print(
                        f"{'mesh_verts':28s}  {status:6s}  {m['corr']:8.5f}  "
                        f"{m['mean']:10.4e}  {m['max']:10.4e}"
                    )
                    if status == "FAIL":
                        reasons = []
                        if not pass_corr:
                            reasons.append(f"corr {m['corr']:.5f} < 0.999")
                        if not pass_mean:
                            reasons.append(f"mean {m['mean']:.4e} > 1e-3")
                        print(f"{'':28s}          ({'; '.join(reasons)})")

                faces_ok = bool(np.array_equal(ref_f, got_f))
                faces_status = "PASS" if faces_ok else "FAIL"
                if not faces_ok:
                    all_pass = False
                print(f"{'mesh_faces':28s}  {faces_status:6s}  identity")

    print()
    if all_pass:
        colored_print("All L2 Swift↔MLX checks PASSED.", GREEN)
    else:
        colored_print("Some L2 Swift↔MLX checks FAILED.", RED)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
