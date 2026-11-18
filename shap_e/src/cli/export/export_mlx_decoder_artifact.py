from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import torch
from requests.exceptions import RequestException
from shap_e.models.download import load_model

from mlx_shap_e.models.rendering.stf_mesh_renderer import save_mlx_decoder_artifact
from mlx_shap_e.offline_tools.runtime_artifact_exports import export_decoder_runtime_families
from mlx_shap_e.utils.cache_recovery import with_corrupt_cache_retry


def _load_decoder_model(cache_dir: Path):
    def _load():
        return load_model("decoder", device=torch.device("cpu"), cache_dir=str(cache_dir))

    def _on_retry(path: Path) -> None:
        print(f"Detected corrupted cache file; deleting and retrying: {path}")

    try:
        return with_corrupt_cache_retry(_load, on_retry=_on_retry, max_attempts=2)
    except RuntimeError as exc:
        msg = str(exc)
        if "should have hash" not in msg or "The file " not in msg:
            raise
        print("Retrying decoder download with a fresh temporary cache directory...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_cache = Path(tmp_dir)
            return with_corrupt_cache_retry(
                lambda: load_model("decoder", device=torch.device("cpu"), cache_dir=str(tmp_cache)),
                on_retry=_on_retry,
                max_attempts=2,
            )


def export_decoder_artifact(*, output_path: Path, cache_dir: Path) -> None:
    output_path = Path(output_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading decoder model...")
    try:
        decoder = _load_decoder_model(cache_dir)
    except RequestException as exc:
        print(f"Failed to fetch decoder artifacts: {exc}")
        print(
            "Tip: ensure network access once to cache vector_decoder files, "
            "or pass --cache-dir pointing to a pre-populated cache."
        )
        raise SystemExit(1)

    print(f"Exporting MLX decoder artifact to {output_path}...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output_path.stem + ".",
            suffix=".npz",
            dir=output_path.parent,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        save_mlx_decoder_artifact(tmp_path, decoder)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None and tmp_path.exists() and tmp_path != output_path:
            tmp_path.unlink()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLX STF decoder mesh artifact (.npz).")
    parser.add_argument(
        "--output",
        default=None,
        help="Output artifact .npz path (defaults to shap_e/outputs/artifacts/mlx_decoder_mesh_v1.npz).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional runtime bundle directory to populate with the decoder_params and renderer families.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    if args.bundle_dir:
        export_decoder_runtime_families(
            bundle_dir=Path(args.bundle_dir).expanduser().resolve(),
            cache_dir=cache_dir,
        )
        return

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else repo_root / "shap_e" / "outputs" / "artifacts" / "mlx_decoder_mesh_v1.npz"
    )
    export_decoder_artifact(output_path=output_path, cache_dir=cache_dir)


if __name__ == "__main__":
    main()
