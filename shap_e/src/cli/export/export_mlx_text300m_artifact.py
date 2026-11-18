from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import torch
from requests.exceptions import RequestException
from shap_e.models.download import load_model

from mlx_shap_e.models.artifacts.text300m_artifact import save_text300m_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.offline_tools.runtime_artifact_exports import export_text300m_runtime_family
from mlx_shap_e.utils.cache_recovery import with_corrupt_cache_retry


def _load_text300m_model(cache_dir: Path):
    def _load():
        return load_model("text300M", device=torch.device("cpu"), cache_dir=str(cache_dir))

    def _on_retry(path: Path) -> None:
        print(f"Detected corrupted cache file; deleting and retrying: {path}")

    return with_corrupt_cache_retry(_load, on_retry=_on_retry, max_attempts=2)


def export_text300m_artifact(*, output_path: Path, cache_dir: Path) -> None:
    output_path = Path(output_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading text300M model...")
    try:
        pt_model = _load_text300m_model(cache_dir)
    except RequestException as exc:
        print(f"Failed to fetch text300M artifacts: {exc}")
        print(
            "Tip: ensure network access once to cache text300M files, "
            "or pass --cache-dir pointing to a pre-populated cache."
        )
        raise SystemExit(1)

    print("Mapping text300M weights to MLX format...")
    mlx_tree = map_text300m_weights(pt_model.state_dict())
    n_ctx = int(getattr(pt_model, "n_ctx", 1024))
    d_latent = int(getattr(pt_model, "d_latent", 1024))

    print(f"Exporting MLX text300M artifact to {output_path}...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output_path.stem + ".",
            suffix=".npz",
            dir=output_path.parent,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        save_text300m_artifact(tmp_path, mlx_tree=mlx_tree, n_ctx=n_ctx, d_latent=d_latent)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None and tmp_path.exists() and tmp_path != output_path:
            tmp_path.unlink()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLX text300M weights to artifact (.npz).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output artifact .npz path (defaults to shap_e/outputs/artifacts/mlx_text300m_v1.npz).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional runtime bundle directory to populate with the text300m family.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    if args.bundle_dir:
        export_text300m_runtime_family(
            bundle_dir=Path(args.bundle_dir).expanduser().resolve(),
            cache_dir=cache_dir,
        )
        return

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else repo_root / "shap_e" / "outputs" / "artifacts" / "mlx_text300m_v1.npz"
    )
    export_text300m_artifact(output_path=output_path, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
