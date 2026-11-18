import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from shap_e.models.download import load_model

from mlx_shap_e.models.embeddings.clip_embedding_artifact import load_clip_embedding_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.text300m import Text300MTransformer


def _slugify(text: str, max_len: int = 40) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "prompt")[:max_len]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLX text300M forward pass with text conditioning.")
    parser.add_argument("--prompt", required=True, help="Text prompt for CLIP conditioning.")
    parser.add_argument("--timestep", type=float, default=500.0, help="Diffusion timestep value.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for latent input tensor.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path (defaults to shap_e/outputs/text300m_<prompt>.npz).",
    )
    parser.add_argument(
        "--parity-check",
        action="store_true",
        help="Also run PyTorch wrapped forward and print parity error metrics.",
    )
    parser.add_argument(
        "--clip-embedding-path",
        default=None,
        help="Path to precomputed CLIP embedding artifact (.npz/.npy).",
    )
    parser.add_argument(
        "--clip-embedding-key",
        default="clip_embedding",
        help="Key used when --clip-embedding-path points to an .npz file.",
    )
    parser.add_argument(
        "--allow-torch-clip-embed-fallback",
        action="store_true",
        help="Allow legacy Torch CLIP embedding from --prompt when no embedding artifact is provided.",
    )
    parser.add_argument(
        "--mlx-clip-text-artifact",
        default=None,
        help=(
            "Path to MLX CLIP text artifact .npz "
            "(default: shap_e/outputs/artifacts/mlx_clip_text_vit_l14_v1.npz)."
        ),
    )
    parser.add_argument(
        "--clip-bpe-path",
        default=None,
        help="Path to CLIP tokenizer merges file bpe_simple_vocab_16e6.txt.gz.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"

    outputs_dir = repo_root / "shap_e" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else outputs_dir / f"text300m_{_slugify(args.prompt)}.npz"
    )
    clip_text_artifact_path = (
        Path(args.mlx_clip_text_artifact).expanduser().resolve()
        if args.mlx_clip_text_artifact
        else repo_root / "shap_e" / "outputs" / "artifacts" / "mlx_clip_text_vit_l14_v1.npz"
    )

    print("Loading PyTorch text300M model from cache...")
    pt_model = load_model("text300M", device=torch.device("cpu"), cache_dir=str(cache_dir))
    pt_model.eval()

    print("Building MLX model and loading mapped weights...")
    mlx_model = Text300MTransformer()
    mlx_tree = map_text300m_weights(pt_model.state_dict())
    mlx_model.update(mlx_tree)

    rng = np.random.default_rng(seed=args.seed)
    x_np = rng.standard_normal((1, 1024, 1024), dtype=np.float32)
    t_np = np.array([args.timestep], dtype=np.float32)

    if args.clip_embedding_path:
        clip_emb = load_clip_embedding_artifact(args.clip_embedding_path, key=args.clip_embedding_key)
        print(f"Loaded CLIP embedding from artifact: {Path(args.clip_embedding_path).expanduser().resolve()}")
    elif clip_text_artifact_path.exists():
        embedder = MlxClipTextEmbedder.from_artifact(
            clip_text_artifact_path,
            tokenizer_bpe_path=args.clip_bpe_path,
        )
        clip_emb = embedder.embed_text([args.prompt]).astype(np.float32)
        print(f"Embedded prompt via MLX CLIP text artifact: {clip_text_artifact_path}")
    elif args.allow_torch_clip_embed_fallback:
        with torch.no_grad():
            clip_emb = pt_model.wrapped.clip.embed_text([args.prompt]).cpu().numpy().astype(np.float32)
        print("Loaded CLIP embedding via legacy Torch path.")
    else:
        raise ValueError(
            "No CLIP embedding source configured. "
            "Provide --clip-embedding-path, or export MLX CLIP text artifact with "
            "shap_e/src/export_mlx_clip_text_artifact.py, or set --allow-torch-clip-embed-fallback."
        )

    y_mx_np = np.array(
        mlx_model(
            mx.array(x_np),
            mx.array(t_np),
            mx.array(clip_emb),
        )
    )

    metrics = {}
    if args.parity_check:
        with torch.no_grad():
            y_pt = pt_model.wrapped(
                torch.from_numpy(x_np),
                torch.from_numpy(t_np),
                embeddings=[torch.from_numpy(clip_emb[0])],
            )
        y_pt_np = y_pt.detach().cpu().numpy()
        diff = y_mx_np - y_pt_np
        metrics = {
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
        }
        print("Parity metrics:")
        for name, value in metrics.items():
            print(f"- {name}: {value:.6e}")

    np.savez_compressed(
        output_path,
        prompt=np.array([args.prompt]),
        timestep=t_np,
        x_input=x_np,
        clip_embedding=clip_emb,
        mlx_output=y_mx_np,
        **{k: np.array([v], dtype=np.float32) for k, v in metrics.items()},
    )

    print(f"Saved run artifact: {output_path}")
    print(f"Output shape: {y_mx_np.shape}")


if __name__ == "__main__":
    main()
