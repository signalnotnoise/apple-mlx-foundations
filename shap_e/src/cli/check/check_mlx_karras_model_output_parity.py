from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.k_diffusion import GaussianToKarrasDenoiser, get_sigmas_karras, sample_heun
from shap_e.models.download import load_config, load_model

from mlx_shap_e.models.embeddings.clip_embedding_artifact import load_clip_embedding_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.diffusion.diffusion_step import split_vector_forward_mlx
from mlx_shap_e.models.text300m import Text300MTransformer, timestep_embedding


def metric_vals(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a - b
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return max_abs, mean_abs, rmse


def metric_line(name: str, a: np.ndarray, b: np.ndarray) -> str:
    max_abs, mean_abs, rmse = metric_vals(a, b)
    return f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, rmse={rmse:.6e}"


def _build_cfg_embeddings(emb_np: np.ndarray, guidance_scale: float) -> np.ndarray:
    if guidance_scale != 1.0 and guidance_scale != 0.0:
        return np.concatenate([emb_np, np.zeros_like(emb_np)], axis=0).astype(np.float32)
    return emb_np.astype(np.float32)


def _to_py_int(x) -> int:
    return int(x.item()) if hasattr(x, "item") else int(x)


def _to_py_float(x) -> float:
    return float(x.item()) if hasattr(x, "item") else float(x)


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _torch_trajectory_states(
    *,
    pt_model,
    diffusion,
    embeddings_np: np.ndarray,
    init_x_np: np.ndarray,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    s_churn: float,
    s_tmin: float,
    s_tmax: float,
    s_noise: float,
    guidance_scale: float,
    clip_denoised: bool,
):
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=torch.device("cpu"))
    x_init = torch.from_numpy(init_x_np)
    model_embeddings_np = _build_cfg_embeddings(embeddings_np, guidance_scale)
    model_kwargs = {"embeddings": [torch.from_numpy(e) for e in model_embeddings_np]}

    denoiser = GaussianToKarrasDenoiser(pt_model, diffusion)

    def denoise_fn(x_t: torch.Tensor, sigma_batch: torch.Tensor) -> torch.Tensor:
        _, pred_xstart = denoiser.denoise(
            x_t,
            sigma_batch,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        return pred_xstart

    if guidance_scale != 0.0 and guidance_scale != 1.0:

        def guided_denoise_fn(x_t: torch.Tensor, sigma_batch: torch.Tensor) -> torch.Tensor:
            x_t_cat = torch.cat([x_t, x_t], dim=0)
            sigma_cat = torch.cat([sigma_batch, sigma_batch], dim=0)
            x0_cat = denoise_fn(x_t_cat, sigma_cat)
            cond_x0, uncond_x0 = torch.chunk(x0_cat, 2, dim=0)
            return uncond_x0 + guidance_scale * (cond_x0 - uncond_x0)

    else:
        guided_denoise_fn = denoise_fn

    states = []
    for obj in sample_heun(
        guided_denoise_fn,
        x_init,
        sigmas,
        progress=False,
        s_churn=s_churn,
        s_tmin=s_tmin,
        s_tmax=s_tmax,
        s_noise=s_noise,
    ):
        if "sigma_hat" not in obj:
            continue
        states.append(
            {
                "step": _to_py_int(obj["i"]) + 1,
                "x": obj["x"].detach().cpu().numpy().astype(np.float32),
                "sigma_hat": _to_py_float(obj["sigma_hat"]),
            }
        )
    return states


def _capture_pt_backbone_outputs(pt_model, x_flat: np.ndarray, t_int: np.ndarray, embeddings_np: np.ndarray) -> dict[str, np.ndarray]:
    bsz = x_flat.shape[0]
    n_ctx = pt_model.n_ctx
    x_nct = x_flat.reshape(bsz, n_ctx, -1).transpose(0, 2, 1)
    t_t = torch.from_numpy(t_int.astype(np.int64))
    x_t = torch.from_numpy(x_nct.astype(np.float32))
    emb_list = [torch.from_numpy(e.astype(np.float32)) for e in embeddings_np]

    captured: dict[str, np.ndarray] = {}
    hooks = []

    def _save(name: str):
        def hook(_module, _inp, out):
            captured[name] = out.detach().cpu().numpy().astype(np.float32)

        return hook

    hooks.append(pt_model.wrapped.ln_pre.register_forward_hook(_save("ln_pre")))
    for i, blk in enumerate(pt_model.wrapped.backbone.resblocks):
        hooks.append(blk.register_forward_hook(_save(f"block_{i:02d}")))

    with torch.no_grad():
        _ = pt_model.wrapped(x_t, t_t, embeddings=emb_list)

    for h in hooks:
        h.remove()

    return captured


def _capture_mlx_backbone_outputs(mlx_model, x_flat: np.ndarray, t_int: np.ndarray, embeddings_np: np.ndarray) -> dict[str, np.ndarray]:
    wrapped = mlx_model.wrapped
    bsz = x_flat.shape[0]
    n_ctx = 1024
    x_nct = x_flat.reshape(bsz, n_ctx, -1).transpose(0, 2, 1)

    x_mx = mx.array(x_nct.astype(np.float32))
    t_mx = mx.array(t_int.astype(np.int32))
    emb_mx = mx.array(embeddings_np.astype(np.float32))

    t_embed = wrapped.time_embed(timestep_embedding(t_mx, wrapped.d_model))
    clip_out = math.sqrt(embeddings_np.shape[-1]) * emb_mx
    clip_embed = wrapped.clip_embed(clip_out)

    cond = [(clip_embed, wrapped.token_cond), (t_embed, wrapped.time_token_cond)]

    h = wrapped.input_proj(mx.transpose(x_mx, (0, 2, 1)))
    for emb, as_token in cond:
        if not as_token:
            h = h + emb[:, None, :]

    if wrapped.use_pos_emb:
        h = h + wrapped.pos_emb[None, :, :]

    extra_tokens = [emb[:, None, :] for emb, as_token in cond if as_token]
    if extra_tokens:
        h = mx.concatenate(extra_tokens + [h], axis=1)

    captured: dict[str, np.ndarray] = {}
    h = wrapped.ln_pre(h)
    captured["ln_pre"] = np.array(h).astype(np.float32)

    for i, blk in enumerate(wrapped.backbone.resblocks.layers):
        h = blk(h)
        captured[f"block_{i:02d}"] = np.array(h).astype(np.float32)

    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description="Karras-step model-output parity with optional layerwise diagnostics.")
    parser.add_argument("--prompt", default="a red chair", help="Text prompt.")
    parser.add_argument("--steps", type=int, default=64, help="Number of Karras steps.")
    parser.add_argument("--sigma-min", type=float, default=1e-3, help="Minimum Karras sigma.")
    parser.add_argument("--sigma-max", type=float, default=160.0, help="Maximum Karras sigma.")
    parser.add_argument("--rho", type=float, default=7.0, help="Karras schedule rho.")
    parser.add_argument("--s-churn", type=float, default=0.0, help="Churn parameter.")
    parser.add_argument("--s-tmin", type=float, default=0.0, help="Churn minimum sigma.")
    parser.add_argument("--s-tmax", type=float, default=float("inf"), help="Churn maximum sigma.")
    parser.add_argument("--s-noise", type=float, default=1.0, help="Noise scale for churn.")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for initial x_T.")
    parser.add_argument(
        "--top-k-steps",
        type=int,
        default=5,
        help="Show top-K worst steps by model_output max abs.",
    )
    parser.add_argument(
        "--layer-analysis-step",
        type=int,
        default=-1,
        help="1-based step for layer analysis; -1 selects the worst step.",
    )
    parser.add_argument(
        "--branch-metrics",
        action="store_true",
        help="When CFG is active, also print cond/uncond and delta branch metrics per step.",
    )
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

    rng = np.random.default_rng(seed=args.seed)
    init_x_np = (rng.standard_normal((1, pt_model.d_latent), dtype=np.float32) * args.sigma_max).astype(np.float32)

    print("Collecting Karras trajectory states from PyTorch reference loop...")
    states = _torch_trajectory_states(
        pt_model=pt_model,
        diffusion=diffusion,
        embeddings_np=emb_np,
        init_x_np=init_x_np,
        steps=args.steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        s_churn=args.s_churn,
        s_tmin=args.s_tmin,
        s_tmax=args.s_tmax,
        s_noise=args.s_noise,
        guidance_scale=args.guidance_scale,
        clip_denoised=True,
    )

    model_embeddings_np = _build_cfg_embeddings(emb_np, args.guidance_scale)
    sigma_to_t = GaussianToKarrasDenoiser(pt_model, diffusion).sigma_to_t

    step_rows = []
    analysis_cache = {}

    print("Per-step raw model-output parity (MLX vs PyTorch)...")
    for st in states:
        sigma_hat = st["sigma_hat"]
        x_step = st["x"]
        c_in = (1.0 / np.sqrt(sigma_hat**2 + 1.0)).astype(np.float32)

        if args.guidance_scale != 0.0 and args.guidance_scale != 1.0:
            x_eval = np.concatenate([x_step, x_step], axis=0).astype(np.float32)
        else:
            x_eval = x_step.astype(np.float32)

        x_in = (x_eval * c_in).astype(np.float32)
        t_scalar = int(sigma_to_t(float(sigma_hat)))
        t_batch = np.full((x_in.shape[0],), t_scalar, dtype=np.int64)

        with torch.no_grad():
            pt_out = (
                pt_model(
                    torch.from_numpy(x_in),
                    torch.from_numpy(t_batch),
                    embeddings=[torch.from_numpy(e) for e in model_embeddings_np],
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        mlx_out = split_vector_forward_mlx(
            mlx_model=mlx_model,
            x_flat=x_in,
            t_int=t_batch,
            embeddings=model_embeddings_np,
            n_ctx=pt_model.n_ctx,
        )

        eps_pt, var_pt = np.split(pt_out, 2, axis=1)
        eps_mx, var_mx = np.split(mlx_out, 2, axis=1)

        out_max, out_mean, out_rmse = metric_vals(mlx_out, pt_out)
        eps_max, _, _ = metric_vals(eps_mx, eps_pt)
        var_max, _, _ = metric_vals(var_mx, var_pt)

        row = {
            "step": st["step"],
            "sigma_hat": sigma_hat,
            "t": t_scalar,
            "out_max": out_max,
            "out_mean": out_mean,
            "out_rmse": out_rmse,
            "eps_max": eps_max,
            "var_max": var_max,
        }
        step_rows.append(row)
        analysis_cache[st["step"]] = {"x_in": x_in, "t_batch": t_batch}

        print(
            f"step={st['step']:03d} sigma_hat={sigma_hat:.6e} t={t_scalar:4d} "
            f"out_max={out_max:.6e} eps_max={eps_max:.6e} var_max={var_max:.6e}"
        )

        if args.branch_metrics and x_in.shape[0] == 2:
            # Row 0 is cond, row 1 is uncond, matching the concatenation order.
            cond_pt, uncond_pt = pt_out[0:1], pt_out[1:2]
            cond_mx, uncond_mx = mlx_out[0:1], mlx_out[1:2]
            delta_pt = cond_pt - uncond_pt
            delta_mx = cond_mx - uncond_mx
            print(
                "  "
                + metric_line("cond_out", cond_mx, cond_pt)
                + " | "
                + metric_line("uncond_out", uncond_mx, uncond_pt)
                + " | "
                + metric_line("delta_out(cond-uncond)", delta_mx, delta_pt)
            )

    worst = sorted(step_rows, key=lambda r: r["out_max"], reverse=True)
    k = max(1, min(args.top_k_steps, len(worst)))
    print(f"\nTop {k} worst steps by model_output max_abs:")
    for r in worst[:k]:
        print(
            f"step={r['step']:03d} sigma_hat={r['sigma_hat']:.6e} t={r['t']:4d} "
            f"out_max={r['out_max']:.6e} eps_max={r['eps_max']:.6e} var_max={r['var_max']:.6e}"
        )

    if args.layer_analysis_step == -1:
        target_step = worst[0]["step"]
    else:
        target_step = args.layer_analysis_step

    if target_step not in analysis_cache:
        raise ValueError(f"layer-analysis-step={target_step} not in sampled steps")

    print(f"\nLayerwise backbone parity at step={target_step:03d}...")
    step_data = analysis_cache[target_step]
    x_in = step_data["x_in"]
    t_batch = step_data["t_batch"]

    pt_layers = _capture_pt_backbone_outputs(pt_model, x_in, t_batch, model_embeddings_np)
    mx_layers = _capture_mlx_backbone_outputs(mlx_model, x_in, t_batch, model_embeddings_np)

    keys = ["ln_pre"] + [f"block_{i:02d}" for i in range(24)]
    for kname in keys:
        if kname not in pt_layers or kname not in mx_layers:
            continue
        p = pt_layers[kname]
        m = mx_layers[kname]
        print(metric_line(kname, m, p))


if __name__ == "__main__":
    main()
