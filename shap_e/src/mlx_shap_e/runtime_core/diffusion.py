from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimeDiffusionConfig:
    """Minimal diffusion state required by the MLX runtime path."""

    alphas_cumprod: np.ndarray
    betas: np.ndarray
    posterior_log_variance_clipped: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray
    num_timesteps: int
    model_mean_type: str = "epsilon"
    channel_biases: np.ndarray | None = None
    channel_scales: np.ndarray | None = None

    @classmethod
    def from_gaussian_diffusion(cls, diffusion: object) -> "RuntimeDiffusionConfig":
        def _optional(name: str) -> np.ndarray | None:
            value = getattr(diffusion, name, None)
            if value is None:
                return None
            return np.asarray(value, dtype=np.float32)

        return cls(
            alphas_cumprod=np.asarray(diffusion.alphas_cumprod, dtype=np.float32),
            betas=np.asarray(diffusion.betas, dtype=np.float32),
            posterior_log_variance_clipped=np.asarray(diffusion.posterior_log_variance_clipped, dtype=np.float32),
            sqrt_recip_alphas_cumprod=np.asarray(diffusion.sqrt_recip_alphas_cumprod, dtype=np.float32),
            sqrt_recipm1_alphas_cumprod=np.asarray(diffusion.sqrt_recipm1_alphas_cumprod, dtype=np.float32),
            posterior_mean_coef1=np.asarray(diffusion.posterior_mean_coef1, dtype=np.float32),
            posterior_mean_coef2=np.asarray(diffusion.posterior_mean_coef2, dtype=np.float32),
            num_timesteps=int(diffusion.num_timesteps),
            model_mean_type=str(getattr(diffusion, "model_mean_type", "epsilon")),
            channel_biases=_optional("channel_biases"),
            channel_scales=_optional("channel_scales"),
        )
