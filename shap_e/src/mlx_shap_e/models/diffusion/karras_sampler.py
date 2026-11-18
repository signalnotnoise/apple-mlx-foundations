from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Optional

import mlx.core as mx
import numpy as np
import torch
from shap_e.diffusion.k_diffusion import get_sigmas_karras

from mlx_shap_e.models.diffusion.diffusion_step import (
    p_mean_variance_learned_range_mlx,
    split_vector_forward_mlx,
)


def _as_mx(x, *, dtype=None) -> mx.array:
    out = x if isinstance(x, mx.array) else mx.array(x)
    return out.astype(dtype) if dtype is not None and out.dtype != dtype else out


def _as_np(x, *, dtype=None) -> np.ndarray:
    out = np.asarray(x)
    return out.astype(dtype) if dtype is not None and out.dtype != dtype else out


def _append_dims(x: np.ndarray, target_dims: int) -> np.ndarray:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def _append_dims_mx(x: mx.array, target_dims: int) -> mx.array:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    out = x
    for _ in range(dims_to_append):
        out = mx.expand_dims(out, axis=-1)
    return out


def _append_zero(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.zeros((1,), dtype=x.dtype)], axis=0)


def _append_zero_mx(x: mx.array) -> mx.array:
    return mx.concatenate([x, mx.zeros((1,), dtype=x.dtype)], axis=0)


def _append_dims_torch(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def get_sigmas_karras_numpy(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> np.ndarray:
    """Construct Karras et al. (2022) sigma schedule and append terminal zero."""
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return _append_zero(sigmas.astype(np.float32))


def get_sigmas_karras_mlx(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> mx.array:
    ramp = mx.linspace(0.0, 1.0, n).astype(mx.float32)
    min_inv_rho = float(sigma_min) ** (1.0 / float(rho))
    max_inv_rho = float(sigma_max) ** (1.0 / float(rho))
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** float(rho)
    return _append_zero_mx(sigmas.astype(mx.float32))


def unscale_channels_numpy(diffusion, x: np.ndarray) -> np.ndarray:
    out = x
    if diffusion.channel_biases is not None:
        out = out - diffusion.channel_biases.reshape((1, -1)).astype(np.float32)
    if diffusion.channel_scales is not None:
        out = out / diffusion.channel_scales.reshape((1, -1)).astype(np.float32)
    return out.astype(np.float32)


def unscale_channels_mlx(
    diffusion,
    x: mx.array,
    *,
    channel_biases_mx: mx.array | None = None,
    channel_scales_mx: mx.array | None = None,
) -> mx.array:
    out = x.astype(mx.float32)
    if diffusion.channel_biases is not None:
        if channel_biases_mx is None:
            channel_biases_mx = mx.array(diffusion.channel_biases.reshape((1, -1)).astype(np.float32))
        out = out - channel_biases_mx
    if diffusion.channel_scales is not None:
        if channel_scales_mx is None:
            channel_scales_mx = mx.array(diffusion.channel_scales.reshape((1, -1)).astype(np.float32))
        out = out / channel_scales_mx
    return out.astype(mx.float32)


@dataclass
class MLXGaussianToKarrasDenoiser:
    mlx_model: object
    diffusion: object
    n_ctx: int = 1024

    def __post_init__(self) -> None:
        alphas = np.asarray(self.diffusion.alphas_cumprod, dtype=np.float64)
        self._alpha_hi = float(alphas[0])
        self._alpha_lo = float(alphas[-1])
        self._last_t = int(self.diffusion.num_timesteps - 1)
        self._alphas_asc = alphas[::-1]
        self._timesteps_asc = np.arange(self.diffusion.num_timesteps, dtype=np.float64)[::-1]
        self._alphas_asc_mx = mx.array(self._alphas_asc.astype(np.float32))
        self._timesteps_asc_mx = mx.array(self._timesteps_asc.astype(np.float32))
        self._channel_biases_mx = (
            mx.array(self.diffusion.channel_biases.reshape((1, -1)).astype(np.float32))
            if self.diffusion.channel_biases is not None
            else None
        )
        self._channel_scales_mx = (
            mx.array(self.diffusion.channel_scales.reshape((1, -1)).astype(np.float32))
            if self.diffusion.channel_scales is not None
            else None
        )

    def _sigma_to_t_numpy(self, sigmas_np: np.ndarray) -> np.ndarray:
        alpha_cumprod = 1.0 / (sigmas_np.astype(np.float64) ** 2 + 1.0)
        t = np.interp(alpha_cumprod, self._alphas_asc, self._timesteps_asc)
        t = np.where(alpha_cumprod > self._alpha_hi, 0, t)
        t = np.where(alpha_cumprod <= self._alpha_lo, self._last_t, t)
        return t.astype(np.int64)

    def sigma_to_t_batch_mx(self, sigmas: mx.array) -> mx.array:
        sigmas_mx = _as_mx(sigmas, dtype=mx.float32)
        alpha_cumprod = (1.0 / (sigmas_mx * sigmas_mx + 1.0)).astype(mx.float32)

        # Find interpolation segment indices as in np.interp with ascending xp.
        ge_mask = alpha_cumprod[:, None] >= self._alphas_asc_mx[None, :]
        idx = mx.sum(ge_mask.astype(mx.int32), axis=1) - 1
        max_idx = self._alphas_asc_mx.shape[0] - 2
        idx = mx.clip(idx, 0, max_idx).astype(mx.int32)

        x0 = self._alphas_asc_mx[idx]
        x1 = self._alphas_asc_mx[idx + 1]
        y0 = self._timesteps_asc_mx[idx]
        y1 = self._timesteps_asc_mx[idx + 1]
        denom = x1 - x0
        frac = mx.where(mx.abs(denom) > 1e-12, (alpha_cumprod - x0) / denom, mx.zeros_like(alpha_cumprod))
        t = y0 + frac * (y1 - y0)

        t = mx.where(alpha_cumprod > self._alpha_hi, mx.array(0.0, dtype=mx.float32), t)
        t = mx.where(alpha_cumprod <= self._alpha_lo, mx.array(float(self._last_t), dtype=mx.float32), t)
        return t.astype(mx.int32)

    def denoise_torch_pmv(
        self,
        x_t: np.ndarray,
        sigmas: np.ndarray,
        embeddings: np.ndarray,
        clip_denoised: bool = True,
    ) -> np.ndarray:
        t_int = self._sigma_to_t_numpy(sigmas)
        c_in = _append_dims(1.0 / np.sqrt(sigmas**2 + 1.0), x_t.ndim).astype(np.float32)
        x_in = (x_t * c_in).astype(np.float32)

        model_output = split_vector_forward_mlx(
            mlx_model=self.mlx_model,
            x_flat=x_in,
            t_int=t_int,
            embeddings=embeddings,
            n_ctx=self.n_ctx,
            return_numpy=True,
        )

        x_in_t = torch.from_numpy(x_in.astype(np.float32))
        t_t = torch.from_numpy(t_int.astype(np.int64))
        model_output_t = torch.from_numpy(model_output.astype(np.float32))

        def model_stub(_x, _t, **kwargs):
            _ = (_x, _t, kwargs)
            return model_output_t

        out = self.diffusion.p_mean_variance(
            model_stub,
            x_in_t,
            t_t,
            clip_denoised=clip_denoised,
            model_kwargs={},
        )
        return out["pred_xstart"].detach().cpu().numpy().astype(np.float32)

    def denoise_mlx(
        self,
        x_t: mx.array,
        sigmas: mx.array,
        embeddings: mx.array,
        clip_denoised: bool = True,
    ) -> mx.array:
        t_int = self.sigma_to_t_batch_mx(sigmas)
        c_in = _append_dims_mx(1.0 / mx.sqrt(sigmas * sigmas + 1.0), x_t.ndim).astype(mx.float32)
        x_in = (x_t * c_in).astype(mx.float32)

        model_output = split_vector_forward_mlx(
            mlx_model=self.mlx_model,
            x_flat=x_in,
            t_int=t_int,
            embeddings=embeddings,
            n_ctx=self.n_ctx,
            return_numpy=False,
        )
        out = p_mean_variance_learned_range_mlx(
            diffusion=self.diffusion,
            x=x_in,
            model_output=model_output,
            t_int=t_int,
            clip_denoised=clip_denoised,
            model_mean_type=getattr(self.diffusion, "model_mean_type", "epsilon"),
        )
        return out["pred_xstart"].astype(mx.float32)


def _to_d(x: np.ndarray, sigma: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    return (x - denoised) / _append_dims(sigma, x.ndim)


def _to_d_mx(x: mx.array, sigma: mx.array, denoised: mx.array) -> mx.array:
    return (x - denoised) / _append_dims_mx(sigma, x.ndim)


def _to_d_torch(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / _append_dims_torch(sigma, x.ndim)


def karras_sample_heun_mlx(
    *,
    diffusion,
    mlx_model,
    embeddings,
    shape: tuple[int, int],
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    guidance_scale: float = 0.0,
    clip_denoised: bool = True,
    seed: int = 0,
    init_x=None,
    n_ctx: int = 1024,
    progress: bool = False,
    return_history: bool = False,
    strict_torch_math: bool = False,
    return_numpy: bool = True,
):
    """MLX denoiser with selectable Karras Heun math backend."""
    rng = np.random.default_rng(seed=seed)
    denoiser = MLXGaussianToKarrasDenoiser(mlx_model=mlx_model, diffusion=diffusion, n_ctx=n_ctx)

    embeddings_np = _as_np(embeddings, dtype=np.float32)
    embeddings_mx = _as_mx(embeddings, dtype=mx.float32)
    init_x_np = _as_np(init_x, dtype=np.float32) if init_x is not None else None

    if guidance_scale != 1.0 and guidance_scale != 0.0:
        model_embeddings_np = np.concatenate([embeddings_np, np.zeros_like(embeddings_np)], axis=0).astype(np.float32)
        model_embeddings_mx = mx.concatenate([embeddings_mx, mx.zeros_like(embeddings_mx)], axis=0).astype(mx.float32)
    else:
        model_embeddings_np = embeddings_np.astype(np.float32)
        model_embeddings_mx = embeddings_mx.astype(mx.float32)

    history_x = []
    history_pred_xstart = []

    if strict_torch_math:
        sigmas_t = get_sigmas_karras(steps, sigma_min, sigma_max, rho=rho, device=torch.device("cpu"))
        sigmas = sigmas_t.detach().cpu().numpy().astype(np.float32)

        if init_x is None:
            x_np = (rng.standard_normal(shape, dtype=np.float32) * sigma_max).astype(np.float32)
        else:
            x_np = init_x_np.copy()

        def denoise_fn(x_t_in: np.ndarray, sigma_batch_in: np.ndarray) -> np.ndarray:
            return denoiser.denoise_torch_pmv(
                x_t=x_t_in.astype(np.float32),
                sigmas=sigma_batch_in.astype(np.float32),
                embeddings=model_embeddings_np,
                clip_denoised=clip_denoised,
            )

        if guidance_scale != 0.0 and guidance_scale != 1.0:

            def guided_denoise_fn(x_t_in: np.ndarray, sigma_batch_in: np.ndarray) -> np.ndarray:
                x_t_cat = np.concatenate([x_t_in, x_t_in], axis=0).astype(np.float32)
                sigma_cat = np.concatenate([sigma_batch_in, sigma_batch_in], axis=0).astype(np.float32)
                x0_cat = denoise_fn(x_t_cat, sigma_cat)
                cond_x0, uncond_x0 = np.split(x0_cat, 2, axis=0)
                return (uncond_x0 + guidance_scale * (cond_x0 - uncond_x0)).astype(np.float32)

        else:
            guided_denoise_fn = denoise_fn

        bsz = shape[0]
        s_in = np.ones((bsz,), dtype=np.float32)
        denoised_np = np.zeros_like(x_np, dtype=np.float32)

        for i in range(len(sigmas_t) - 1):
            sigma_i = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])
            gamma = min(s_churn / (len(sigmas_t) - 1), np.sqrt(2.0) - 1.0) if s_tmin <= sigma_i <= s_tmax else 0.0

            sigma_hat = sigma_i * (gamma + 1.0)
            if gamma > 0.0:
                eps = (rng.standard_normal(shape, dtype=np.float32) * s_noise).astype(np.float32)
                x_np = x_np + eps * np.sqrt(max(sigma_hat**2 - sigma_i**2, 0.0))

            sigma_hat_batch = (np.array(sigma_hat, dtype=np.float32) * s_in).astype(np.float32)
            denoised_np = guided_denoise_fn(x_np, sigma_hat_batch)
            d_np = _to_d(x_np, sigma_hat_batch, denoised_np).astype(np.float32)

            if return_history:
                history_x.append(unscale_channels_numpy(diffusion, x_np.astype(np.float32)))
                history_pred_xstart.append(
                    unscale_channels_numpy(diffusion, denoised_np.astype(np.float32))
                )

            if progress:
                print(f"step={i + 1}/{len(sigmas) - 1} sigma={sigma_i:.6f} -> {sigma_next:.6f}")

            dt = sigma_next - sigma_hat
            if sigma_next == 0.0:
                x_np = x_np + d_np * dt
            else:
                x_2_np = x_np + d_np * dt
                sigma_next_batch = (np.array(sigma_next, dtype=np.float32) * s_in).astype(np.float32)
                denoised_2_np = guided_denoise_fn(x_2_np, sigma_next_batch)
                d_2_np = _to_d(x_2_np, sigma_next_batch, denoised_2_np).astype(np.float32)
                d_prime_np = ((d_np + d_2_np) / 2.0).astype(np.float32)
                x_np = x_np + d_prime_np * dt

        if return_history:
            history_x.append(unscale_channels_numpy(diffusion, x_np.astype(np.float32)))
            history_pred_xstart.append(
                unscale_channels_numpy(diffusion, denoised_np.astype(np.float32))
            )

        out = {
            "x": unscale_channels_numpy(diffusion, x_np.astype(np.float32)),
            "pred_xstart": unscale_channels_numpy(diffusion, denoised_np.astype(np.float32)),
            "sigmas": sigmas,
        }
    else:
        print("Using MLX for Karras Heun math backend (non-parity-locked, may differ from PyTorch version).")
        sigmas_mx = get_sigmas_karras_mlx(steps, sigma_min, sigma_max, rho=rho)
        sigmas = sigmas_mx

        if init_x is None:
            x = mx.array((rng.standard_normal(shape, dtype=np.float32) * sigma_max).astype(np.float32))
        else:
            x = mx.array(init_x_np.copy())

        if guidance_scale != 0.0 and guidance_scale != 1.0:

            def guided_denoise_fn_mx(x_t_in: mx.array, sigma_batch_in: mx.array) -> mx.array:
                x_t_cat = mx.concatenate([x_t_in, x_t_in], axis=0)
                sigma_cat = mx.concatenate([sigma_batch_in, sigma_batch_in], axis=0)
                x0_cat = denoiser.denoise_mlx(
                    x_t=x_t_cat,
                    sigmas=sigma_cat,
                    embeddings=model_embeddings_mx,
                    clip_denoised=clip_denoised,
                )
                cond_x0, uncond_x0 = mx.split(x0_cat, 2, axis=0)
                return (uncond_x0 + guidance_scale * (cond_x0 - uncond_x0)).astype(mx.float32)

        else:

            def guided_denoise_fn_mx(x_t_in: mx.array, sigma_batch_in: mx.array) -> mx.array:
                return denoiser.denoise_mlx(
                    x_t=x_t_in,
                    sigmas=sigma_batch_in,
                    embeddings=model_embeddings_mx,
                    clip_denoised=clip_denoised,
                ).astype(mx.float32)

        bsz = shape[0]
        s_in = mx.ones((bsz,), dtype=mx.float32)
        denoised = mx.zeros_like(x)

        sigmas_np = np.asarray(sigmas, dtype=np.float32)
        n_sigmas = int(sigmas_np.shape[0])
        for i in range(n_sigmas - 1):
            sigma_i = float(sigmas_np[i])
            sigma_next = float(sigmas_np[i + 1])
            gamma = min(s_churn / (n_sigmas - 1), sqrt(2.0) - 1.0) if s_tmin <= sigma_i <= s_tmax else 0.0

            sigma_hat = sigma_i * (gamma + 1.0)
            if gamma > 0.0:
                eps = mx.array((rng.standard_normal(shape, dtype=np.float32) * s_noise).astype(np.float32))
                x = x + eps * mx.sqrt(mx.array(max(sigma_hat**2 - sigma_i**2, 0.0), dtype=mx.float32))

            sigma_hat_batch = mx.array(sigma_hat, dtype=mx.float32) * s_in
            denoised = guided_denoise_fn_mx(x, sigma_hat_batch)
            d = _to_d_mx(x, sigma_hat_batch, denoised)

            if return_history:
                history_x.append(
                    unscale_channels_mlx(
                        diffusion,
                        x,
                        channel_biases_mx=denoiser._channel_biases_mx,
                        channel_scales_mx=denoiser._channel_scales_mx,
                    )
                )
                history_pred_xstart.append(
                    unscale_channels_mlx(
                        diffusion,
                        denoised,
                        channel_biases_mx=denoiser._channel_biases_mx,
                        channel_scales_mx=denoiser._channel_scales_mx,
                    )
                )

            if progress:
                print(f"step={i + 1}/{n_sigmas - 1} sigma={sigma_i:.6f} -> {sigma_next:.6f}")

            dt = sigma_next - sigma_hat
            if sigma_next == 0.0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                sigma_next_batch = mx.array(sigma_next, dtype=mx.float32) * s_in
                denoised_2 = guided_denoise_fn_mx(x_2, sigma_next_batch)
                d_2 = _to_d_mx(x_2, sigma_next_batch, denoised_2)
                x = x + ((d + d_2) / 2.0) * dt

        if return_history:
            history_x.append(
                unscale_channels_mlx(
                    diffusion,
                    x,
                    channel_biases_mx=denoiser._channel_biases_mx,
                    channel_scales_mx=denoiser._channel_scales_mx,
                )
            )
            history_pred_xstart.append(
                unscale_channels_mlx(
                    diffusion,
                    denoised,
                    channel_biases_mx=denoiser._channel_biases_mx,
                    channel_scales_mx=denoiser._channel_scales_mx,
                )
            )

        out = {
            "x": unscale_channels_mlx(
                diffusion,
                x,
                channel_biases_mx=denoiser._channel_biases_mx,
                channel_scales_mx=denoiser._channel_scales_mx,
            ),
            "pred_xstart": unscale_channels_mlx(
                diffusion,
                denoised,
                channel_biases_mx=denoiser._channel_biases_mx,
                channel_scales_mx=denoiser._channel_scales_mx,
            ),
            "sigmas": sigmas,
        }

    if return_history:
        if strict_torch_math:
            out["history_x"] = np.stack(history_x, axis=0).astype(np.float32)
            out["history_pred_xstart"] = np.stack(history_pred_xstart, axis=0).astype(np.float32)
        else:
            out["history_x"] = mx.stack(history_x, axis=0)
            out["history_pred_xstart"] = mx.stack(history_pred_xstart, axis=0)

    if return_numpy:
        return {k: _as_np(v, dtype=np.float32) for k, v in out.items()}
    return out
