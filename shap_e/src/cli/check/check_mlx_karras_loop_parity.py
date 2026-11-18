from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.k_diffusion import GaussianToKarrasDenoiser, get_sigmas_karras, sample_heun
from shap_e.models.download import load_config, load_model

from mlx_shap_e.models.embeddings.clip_embedding_artifact import load_clip_embedding_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.diffusion.karras_sampler import karras_sample_heun_mlx
from mlx_shap_e.models.text300m import Text300MTransformer

from cli.check.color import colored_metric


def metric(name: str, a: np.ndarray, b: np.ndarray) -> str:
    return colored_metric(name, a, b)


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def _torch_karras_heun_reference(
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
) -> dict[str, np.ndarray]:
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=torch.device("cpu"))
    x_init = torch.from_numpy(init_x_np)

    if guidance_scale != 1.0 and guidance_scale != 0.0:
        model_embeddings_np = np.concatenate([embeddings_np, np.zeros_like(embeddings_np)], axis=0)
    else:
        model_embeddings_np = embeddings_np

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

    last = None
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
        last = obj

    assert last is not None
    x_unscaled = diffusion.unscale_channels(last["x"]).detach().cpu().numpy().astype(np.float32)
    pred_xstart_unscaled = diffusion.unscale_channels(last["pred_xstart"]).detach().cpu().numpy().astype(np.float32)

    return {
        "x": x_unscaled,
        "pred_xstart": pred_xstart_unscaled,
        "sigmas": sigmas.detach().cpu().numpy().astype(np.float32),
    }


def _torch_karras_heun_progressive(
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
) -> list[dict[str, np.ndarray]]:
    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=torch.device("cpu"))
    x_init = torch.from_numpy(init_x_np)

    if guidance_scale != 1.0 and guidance_scale != 0.0:
        model_embeddings_np = np.concatenate([embeddings_np, np.zeros_like(embeddings_np)], axis=0)
    else:
        model_embeddings_np = embeddings_np

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

    history: list[dict[str, np.ndarray]] = []
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
        history.append(
            {
                "x": diffusion.unscale_channels(obj["x"]).detach().cpu().numpy().astype(np.float32),
                "pred_xstart": diffusion.unscale_channels(obj["pred_xstart"]).detach().cpu().numpy().astype(np.float32),
            }
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full MLX Karras Heun loop vs PyTorch reference.")
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
        "--integrator-math",
        choices=["torch", "mlx"],
        default="torch",
        help="Karras schedule/integrator math backend for MLX loop.",
    )
    parser.add_argument(
        "--report-steps",
        action="store_true",
        help="Also print per-step max-abs drift vs Torch reference.",
    )
    parser.add_argument(
        "--compare-torch-vs-mlx-timing",
        action="store_true",
        help="Benchmark both MLX loop backends (integrator_math=torch and integrator_math=mlx) and compare outputs.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=1,
        help="Number of runs per backend when --compare-torch-vs-mlx-timing is enabled.",
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

    print("Running PyTorch Karras Heun reference loop...")
    out_pt = _torch_karras_heun_reference(
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

    print("Running MLX Karras Heun loop...")
    out_mlx = karras_sample_heun_mlx(
        diffusion=diffusion,
        mlx_model=mlx_model,
        embeddings=emb_np,
        shape=(1, pt_model.d_latent),
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
        seed=args.seed,
        init_x=init_x_np,
        n_ctx=pt_model.n_ctx,
        progress=False,
        return_history=args.report_steps,
        strict_torch_math=(args.integrator_math == "torch"),
    )

    print("Karras loop parity metrics (MLX vs PyTorch):")
    print(metric("x_final", out_mlx["x"], out_pt["x"]))
    print(metric("pred_xstart", out_mlx["pred_xstart"], out_pt["pred_xstart"]))
    print(metric("sigmas", out_mlx["sigmas"], out_pt["sigmas"]))

    if args.compare_torch_vs_mlx_timing:
        repeats = max(1, int(args.timing_repeats))
        print(f"Benchmarking integrator backends (repeats={repeats})...")

        def _run_backend(strict_torch_math: bool) -> tuple[dict[str, np.ndarray], float]:
            total = 0.0
            last_out: dict[str, np.ndarray] | None = None
            for _ in range(repeats):
                t0 = time.perf_counter()
                last_out = karras_sample_heun_mlx(
                    diffusion=diffusion,
                    mlx_model=mlx_model,
                    embeddings=emb_np,
                    shape=(1, pt_model.d_latent),
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
                    seed=args.seed,
                    init_x=init_x_np,
                    n_ctx=pt_model.n_ctx,
                    progress=False,
                    return_history=False,
                    strict_torch_math=strict_torch_math,
                )
                total += time.perf_counter() - t0
            assert last_out is not None
            return last_out, total / repeats

        out_math_torch, avg_torch = _run_backend(strict_torch_math=True)
        out_math_mlx, avg_mlx = _run_backend(strict_torch_math=False)

        print("Timing (avg wall time):")
        print(f"- integrator_math=torch: {avg_torch*1000.0:.2f} ms")
        print(f"- integrator_math=mlx:   {avg_mlx*1000.0:.2f} ms")
        if avg_mlx > 0:
            print(f"- speed ratio (torch/mlx): {avg_torch / avg_mlx:.3f}x")

        print("Output drift (integrator_math=mlx vs integrator_math=torch):")
        print(metric("x_final", out_math_mlx["x"], out_math_torch["x"]))
        print(metric("pred_xstart", out_math_mlx["pred_xstart"], out_math_torch["pred_xstart"]))
        print(metric("sigmas", out_math_mlx["sigmas"], out_math_torch["sigmas"]))

    if args.report_steps:
        print("Computing per-step Torch/MLX drift...")
        torch_hist = _torch_karras_heun_progressive(
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
        mlx_hist_x = out_mlx["history_x"]
        mlx_hist_x0 = out_mlx["history_pred_xstart"]
        n = min(len(torch_hist), int(mlx_hist_x.shape[0]))
        for i in range(n):
            pt_x = torch_hist[i]["x"]
            pt_x0 = torch_hist[i]["pred_xstart"]
            mx_x = mlx_hist_x[i]
            mx_x0 = mlx_hist_x0[i]
            x_max = float(np.max(np.abs(mx_x - pt_x)))
            x0_max = float(np.max(np.abs(mx_x0 - pt_x0)))
            print(f"step={i + 1:03d} x_max_abs={x_max:.6e} pred_xstart_max_abs={x0_max:.6e}")


if __name__ == "__main__":
    main()
