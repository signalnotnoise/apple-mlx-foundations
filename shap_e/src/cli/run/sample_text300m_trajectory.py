import argparse
import math
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


def _cosine_betas(num_timesteps: int, s: float = 0.008) -> np.ndarray:
    def alpha_bar(t: float) -> float:
        return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return np.array(betas, dtype=np.float64)


def _build_step_schedule(total_timesteps: int, sampling_steps: int) -> np.ndarray:
    raw = np.linspace(total_timesteps - 1, 0, sampling_steps)
    steps = np.unique(np.round(raw).astype(np.int32))[::-1]
    if steps[-1] != 0:
        steps = np.append(steps, 0)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text300M latent trajectory with MLX DDIM updates.")
    parser.add_argument("--prompt", required=True, help="Text prompt for CLIP conditioning.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for initial Gaussian noise.")
    parser.add_argument("--sampling-steps", type=int, default=64, help="Number of DDIM sampling steps.")
    parser.add_argument("--total-timesteps", type=int, default=1000, help="Training diffusion horizon.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=8,
        help="Store trajectory snapshots every N sampling steps (plus final step).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path (defaults to shap_e/outputs/trajectory_<prompt>.npz).",
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
        else outputs_dir / f"trajectory_{_slugify(args.prompt)}.npz"
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
    mlx_model.update(map_text300m_weights(pt_model.state_dict()))

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

    betas = _cosine_betas(args.total_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)

    step_schedule = _build_step_schedule(args.total_timesteps, args.sampling_steps)

    rng = np.random.default_rng(seed=args.seed)
    x_t = mx.array(rng.standard_normal((1, 1024, 1024), dtype=np.float32))

    snapshots = []
    snapshot_ts = []

    print(f"Sampling with {len(step_schedule)} scheduled timesteps...")
    for step_idx, t_int in enumerate(step_schedule):
        t_batch = mx.array([float(t_int)], dtype=mx.float32)
        model_out = mlx_model(x_t, t_batch, mx.array(clip_emb))

        eps_pred = model_out[:, :1024, :]

        alpha_bar_t = float(alphas_cumprod[t_int])
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(max(1.0 - alpha_bar_t, 1e-12))

        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / max(sqrt_alpha_bar_t, 1e-12)

        if t_int == 0 or step_idx == len(step_schedule) - 1:
            x_prev = x0_pred
        else:
            t_prev = int(step_schedule[step_idx + 1])
            alpha_bar_prev = float(alphas_cumprod[t_prev])
            x_prev = math.sqrt(alpha_bar_prev) * x0_pred + math.sqrt(max(1.0 - alpha_bar_prev, 1e-12)) * eps_pred

        if (step_idx % max(args.save_every, 1) == 0) or (t_int == 0) or (step_idx == len(step_schedule) - 1):
            snapshots.append(np.array(x_prev).astype(np.float32))
            snapshot_ts.append(int(t_int))

        x_t = x_prev

    final_latent = np.array(x_t).astype(np.float32)
    final_flat = final_latent.reshape(1, -1)

    np.savez_compressed(
        output_path,
        prompt=np.array([args.prompt]),
        seed=np.array([args.seed], dtype=np.int32),
        sampling_steps=np.array([args.sampling_steps], dtype=np.int32),
        total_timesteps=np.array([args.total_timesteps], dtype=np.int32),
        step_schedule=step_schedule.astype(np.int32),
        snapshot_timesteps=np.array(snapshot_ts, dtype=np.int32),
        clip_embedding=clip_emb,
        final_latent=final_latent,
        final_latent_flat=final_flat,
        latent_snapshots=np.stack(snapshots, axis=0),
    )

    print(f"Saved trajectory artifact: {output_path}")
    print(f"Final latent shape: {final_latent.shape}")
    print(f"Saved {len(snapshots)} snapshots at timesteps: {snapshot_ts}")


if __name__ == "__main__":
    main()
