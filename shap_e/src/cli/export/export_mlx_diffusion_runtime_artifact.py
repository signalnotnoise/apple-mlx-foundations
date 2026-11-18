from __future__ import annotations

import argparse
from pathlib import Path

from mlx_shap_e.offline_tools.runtime_artifact_exports import export_diffusion_runtime_family


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the diffusion runtime family into a manifest-based MLX runtime bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Runtime bundle directory to populate with the diffusion family.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    export_diffusion_runtime_family(
        bundle_dir=Path(args.bundle_dir).expanduser().resolve(),
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
