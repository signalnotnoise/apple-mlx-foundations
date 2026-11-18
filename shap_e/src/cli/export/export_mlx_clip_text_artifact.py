from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from mlx_shap_e.offline_tools.runtime_artifact_exports import export_clip_text_runtime_family


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def export_clip_text_artifact(
    *,
    output_path: Path,
    cache_dir: Path,
    clip_name: str = "ViT-L/14",
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import: clip depends on torch/torchvision.
    import clip

    print(f"Loading CLIP model: {clip_name}")
    model, _ = clip.load(clip_name, device=torch.device("cpu"), download_root=str(cache_dir))
    state = model.state_dict()

    n_layers = len([k for k in state.keys() if k.startswith("transformer.resblocks.") and k.endswith(".ln_1.weight")])
    payload: dict[str, np.ndarray] = {
        "artifact_type": np.array(["mlx_clip_text_vit_l14_v1"]),
        "meta_context_length": np.array([int(model.context_length)], dtype=np.int32),
        "meta_transformer_width": np.array([int(model.transformer.width)], dtype=np.int32),
        "meta_transformer_heads": np.array([int(model.transformer.width // 64)], dtype=np.int32),
        "meta_transformer_layers": np.array([int(n_layers)], dtype=np.int32),
        "token_embedding.weight": state["token_embedding.weight"].detach().cpu().numpy().astype(np.float32),
        "positional_embedding": state["positional_embedding"].detach().cpu().numpy().astype(np.float32),
        "ln_final.weight": state["ln_final.weight"].detach().cpu().numpy().astype(np.float32),
        "ln_final.bias": state["ln_final.bias"].detach().cpu().numpy().astype(np.float32),
        "text_projection": state["text_projection"].detach().cpu().numpy().astype(np.float32),
    }

    for i in range(n_layers):
        prefix = f"transformer.resblocks.{i}"
        for name in (
            "ln_1.weight",
            "ln_1.bias",
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ):
            key = f"{prefix}.{name}"
            payload[key] = state[key].detach().cpu().numpy().astype(np.float32)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output_path.stem + ".",
            suffix=".npz",
            dir=output_path.parent,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path is not None and tmp_path.exists() and tmp_path != output_path:
            tmp_path.unlink()
    print(f"Saved MLX CLIP text artifact: {output_path}")
    print(f"layers={n_layers}, width={int(model.transformer.width)}, context={int(model.context_length)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CLIP ViT-L/14 text encoder weights to MLX artifact.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output artifact .npz path (defaults to shap_e/outputs/artifacts/mlx_clip_text_vit_l14_v1.npz).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="CLIP cache/download directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--clip-name",
        default="ViT-L/14",
        choices=["ViT-L/14"],
        help="CLIP model variant to export.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional runtime bundle directory to populate with the clip_text family.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    if args.bundle_dir:
        export_clip_text_runtime_family(
            bundle_dir=_resolve_user_path(args.bundle_dir, repo_root=repo_root),
            cache_dir=cache_dir,
            clip_name=args.clip_name,
        )
        return

    output_path = (
        _resolve_user_path(args.output, repo_root=repo_root)
        if args.output
        else repo_root / "shap_e" / "outputs" / "artifacts" / "mlx_clip_text_vit_l14_v1.npz"
    )
    export_clip_text_artifact(output_path=output_path, cache_dir=cache_dir, clip_name=args.clip_name)


if __name__ == "__main__":
    main()
