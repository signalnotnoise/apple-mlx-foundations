from __future__ import annotations

from typing import Dict

import mlx.core as mx
import numpy as np


def _to_mx(x, *, dtype=None) -> mx.array:
    out = mx.array(x)
    return out.astype(dtype) if dtype is not None and out.dtype != dtype else out


def split_vector_forward_mlx(
    mlx_model,
    x_flat,
    t_int,
    embeddings,
    n_ctx: int = 1024,
    return_numpy: bool = True,
):
    """Run MLX wrapped model in SplitVectorDiffusion layout and return flattened [eps,var]."""
    if x_flat.ndim != 2:
        raise ValueError(f"x_flat must be 2D [B, D], got {x_flat.shape}")

    bsz = x_flat.shape[0]
    x_flat = _to_mx(x_flat, dtype=mx.float32)
    t_int = _to_mx(t_int, dtype=mx.int32)
    embeddings = _to_mx(embeddings, dtype=mx.float32)

    h = mx.transpose(mx.reshape(x_flat, (bsz, n_ctx, -1)), (0, 2, 1))  # [B, C, T]

    y = mlx_model(
        h,
        t_int,
        embeddings,
    )

    eps, var = mx.split(y, 2, axis=1)
    eps_flat = mx.reshape(mx.transpose(eps, (0, 2, 1)), (bsz, -1))
    var_flat = mx.reshape(mx.transpose(var, (0, 2, 1)), (bsz, -1))
    out = mx.concatenate([eps_flat, var_flat], axis=1).astype(mx.float32)
    if return_numpy:
        return np.asarray(out, dtype=np.float32)
    return out


def _extract_into_mx(arr, timesteps, broadcast_shape) -> mx.array:
    """MLX equivalent of shap_e.diffusion.gaussian_diffusion._extract_into_tensor."""
    arr_mx = _to_mx(arr, dtype=mx.float32)
    t_mx = _to_mx(timesteps, dtype=mx.int32)
    res = arr_mx[t_mx]
    while res.ndim < len(broadcast_shape):
        res = mx.expand_dims(res, axis=-1)
    return mx.broadcast_to(res, broadcast_shape).astype(mx.float32)


def p_mean_variance_learned_range_mlx(
    *,
    diffusion,
    x,
    model_output,
    t_int,
    model_mean_type: str = "epsilon",
    clip_denoised: bool = True,
) -> Dict[str, mx.array]:
    """MLX mirror of GaussianDiffusion.p_mean_variance for var_type=learned_range."""
    x = _to_mx(x, dtype=mx.float32)
    model_output = _to_mx(model_output, dtype=mx.float32)
    t_int = _to_mx(t_int, dtype=mx.int32)

    bsz, channels = x.shape[:2]
    if model_output.shape[1] != channels * 2:
        raise ValueError(
            f"model_output second dim must be 2*C ({2*channels}), got {model_output.shape[1]}"
        )

    eps, model_var_values = mx.split(model_output, 2, axis=1)

    min_log = _extract_into_mx(diffusion.posterior_log_variance_clipped, t_int, x.shape)
    max_log = _extract_into_mx(np.log(diffusion.betas), t_int, x.shape)
    frac = (model_var_values + 1.0) / 2.0
    model_log_variance = frac * max_log + (1.0 - frac) * min_log
    model_variance = mx.exp(model_log_variance)

    if model_mean_type == "x_start":
        pred_xstart = eps
        if clip_denoised:
            pred_xstart = mx.clip(pred_xstart, -1.0, 1.0)
    elif model_mean_type == "epsilon":
        pred_xstart = (
            _extract_into_mx(diffusion.sqrt_recip_alphas_cumprod, t_int, x.shape) * x
            - _extract_into_mx(diffusion.sqrt_recipm1_alphas_cumprod, t_int, x.shape) * eps
        )
        if clip_denoised:
            pred_xstart = mx.clip(pred_xstart, -1.0, 1.0)
    else:
        raise NotImplementedError(f"Unsupported model_mean_type: {model_mean_type}")

    model_mean = (
        _extract_into_mx(diffusion.posterior_mean_coef1, t_int, x.shape) * pred_xstart
        + _extract_into_mx(diffusion.posterior_mean_coef2, t_int, x.shape) * x
    )

    return {
        "mean": model_mean.astype(mx.float32),
        "variance": model_variance.astype(mx.float32),
        "log_variance": model_log_variance.astype(mx.float32),
        "pred_xstart": pred_xstart.astype(mx.float32),
    }


def p_mean_variance_learned_range_numpy(
    *,
    diffusion,
    x,
    model_output,
    t_int,
    model_mean_type: str = "epsilon",
    clip_denoised: bool = True,
) -> Dict[str, np.ndarray]:
    out = p_mean_variance_learned_range_mlx(
        diffusion=diffusion,
        x=x,
        model_output=model_output,
        t_int=t_int,
        model_mean_type=model_mean_type,
        clip_denoised=clip_denoised,
    )
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
