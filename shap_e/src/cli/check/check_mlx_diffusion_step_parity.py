import argparse
from pathlib import Path

import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_config, load_model

from mlx_shap_e.models.embeddings.clip_embedding_artifact import load_clip_embedding_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.diffusion.diffusion_step import (
    p_mean_variance_learned_range_numpy,
    split_vector_forward_mlx,
)
from mlx_shap_e.models.text300m import Text300MTransformer


def metric(name: str, a: np.ndarray, b: np.ndarray) -> str:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rmse={rmse:.6e}"


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single diffusion step parity (MLX vs PyTorch).")
    parser.add_argument("--prompt", default="a red chair", help="Text prompt.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
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
    clip_text_artifact_path = (
        _resolve_user_path(args.mlx_clip_text_artifact, repo_root=repo_root)
        if args.mlx_clip_text_artifact
        else repo_root / "shap_e" / "outputs" / "artifacts" / "mlx_clip_text_vit_l14_v1.npz"
    )

    print("Loading PyTorch text300M model + diffusion config...")
    pt_model = load_model("text300M", device=torch.device("cpu"), cache_dir=str(cache_dir))
    pt_model.eval()
    diffusion = diffusion_from_config(load_config("diffusion", cache_dir=str(cache_dir)))

    print("Building MLX text300M model and loading mapped weights...")
    mlx_model = Text300MTransformer()
    mlx_model.update(map_text300m_weights(pt_model.state_dict()))

    rng = np.random.default_rng(seed=11)
    x_np = rng.standard_normal((1, pt_model.d_latent), dtype=np.float32)
    t_np = np.array([500], dtype=np.int64)

    if args.clip_embedding_path:
        emb_np = load_clip_embedding_artifact(args.clip_embedding_path, key=args.clip_embedding_key)
        print(f"Loaded CLIP embedding from artifact: {Path(args.clip_embedding_path).expanduser().resolve()}")
    elif clip_text_artifact_path.exists():
        embedder = MlxClipTextEmbedder.from_artifact(
            clip_text_artifact_path,
            tokenizer_bpe_path=args.clip_bpe_path,
        )
        emb_np = embedder.embed_text([args.prompt]).astype(np.float32)
        print(f"Embedded prompt via MLX CLIP text artifact: {clip_text_artifact_path}")
    elif args.allow_torch_clip_embed_fallback:
        with torch.no_grad():
            emb_np = (
                pt_model.wrapped.clip.embed_text([args.prompt])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        print("Loaded CLIP embedding via legacy Torch path.")
    else:
        raise ValueError(
            "No CLIP embedding source configured. "
            "Provide --clip-embedding-path, or export MLX CLIP text artifact with "
            "shap_e/src/export_mlx_clip_text_artifact.py, or set --allow-torch-clip-embed-fallback."
        )

    print("Running PyTorch diffusion p_mean_variance (reference)...")
    with torch.no_grad():
        out_pt = diffusion.p_mean_variance(
            pt_model,
            torch.from_numpy(x_np),
            torch.from_numpy(t_np),
            clip_denoised=True,
            model_kwargs={"embeddings": [torch.from_numpy(emb_np[0])]},
        )
        model_output_pt = (
            pt_model(
                torch.from_numpy(x_np),
                torch.from_numpy(t_np),
                embeddings=[torch.from_numpy(emb_np[0])],
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    out_pt_np = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out_pt.items() if isinstance(v, torch.Tensor)}

    print("Running MLX split-vector forward + mirrored diffusion step...")
    model_output_mlx = split_vector_forward_mlx(
        mlx_model=mlx_model,
        x_flat=x_np,
        t_int=t_np,
        embeddings=emb_np,
        n_ctx=pt_model.n_ctx,
    )
    out_mlx_np = p_mean_variance_learned_range_numpy(
        diffusion=diffusion,
        x=x_np,
        model_output=model_output_mlx,
        t_int=t_np,
        model_mean_type=diffusion.model_mean_type,
        clip_denoised=True,
    )
    out_manual_pt_np = p_mean_variance_learned_range_numpy(
        diffusion=diffusion,
        x=x_np,
        model_output=model_output_pt,
        t_int=t_np,
        model_mean_type=diffusion.model_mean_type,
        clip_denoised=True,
    )

    print("Diagnostic parity:")
    print(metric("model_output (MLX vs PT)", model_output_mlx, model_output_pt))
    print(metric("manual(mean) using PT output vs diffusion.p_mean_variance", out_manual_pt_np["mean"], out_pt_np["mean"]))
    print(
        metric(
            "manual(pred_xstart) using PT output vs diffusion.p_mean_variance",
            out_manual_pt_np["pred_xstart"],
            out_pt_np["pred_xstart"],
        )
    )

    print("Diffusion-step parity metrics (MLX vs PyTorch):")
    print(metric("mean", out_mlx_np["mean"], out_pt_np["mean"]))
    print(metric("pred_xstart", out_mlx_np["pred_xstart"], out_pt_np["pred_xstart"]))
    print(metric("log_variance", out_mlx_np["log_variance"], out_pt_np["log_variance"]))
    print(metric("variance", out_mlx_np["variance"], out_pt_np["variance"]))


if __name__ == "__main__":
    main()
