from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from shap_e.models.download import load_model

from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from cli.check.color import colored_metric


def metric(name: str, a: np.ndarray, b: np.ndarray) -> str:
    return colored_metric(name, a, b)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "shap_e_model_cache"

    print("Loading decoder model...")
    decoder = load_model("decoder", device=torch.device("cpu"), cache_dir=str(cache_dir))
    decoder.eval()

    print("Building MLX decoder-params projection...")
    mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)

    rng = np.random.default_rng(seed=23)
    z_np = rng.standard_normal((1, decoder.d_latent), dtype=np.float32)

    with torch.no_grad():
        params_pt = decoder.bottleneck_to_params(torch.from_numpy(z_np))
    params_mlx = mlx_decoder.bottleneck_to_params_numpy(z_np)

    print("Decoder params parity metrics (MLX vs PyTorch):")
    for k in params_pt.keys():
        pt_np = params_pt[k].detach().cpu().numpy().astype(np.float32)
        mlx_np = params_mlx[k]
        print(metric(k, mlx_np, pt_np))


if __name__ == "__main__":
    main()
