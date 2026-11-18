from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from shap_e.models.download import load_model

from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.text300m import Text300MTransformer


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "shap_e_model_cache"

    print("Loading PyTorch text300M model...")
    pt_model = load_model("text300M", device=torch.device("cpu"), cache_dir=str(cache_dir))
    pt_model.eval()

    print("Initializing MLX model and loading mapped weights...")
    mlx_model = Text300MTransformer()
    mlx_tree = map_text300m_weights(pt_model.state_dict())
    mlx_model.update(mlx_tree)

    rng = np.random.default_rng(seed=7)
    x_np = rng.standard_normal((1, 1024, 1024), dtype=np.float32)
    t_np = rng.uniform(0.0, 1000.0, size=(1,)).astype(np.float32)
    emb_np = rng.standard_normal((1, 768), dtype=np.float32)

    x_torch = torch.from_numpy(x_np)
    t_torch = torch.from_numpy(t_np)
    emb_torch_list = [torch.from_numpy(emb_np[0])]

    with torch.no_grad():
        y_pt = pt_model.wrapped(x_torch, t_torch, embeddings=emb_torch_list)
    y_pt_np = y_pt.detach().cpu().numpy()

    x_mx = mx.array(x_np)
    t_mx = mx.array(t_np)
    emb_mx = mx.array(emb_np)
    y_mx = mlx_model(x_mx, t_mx, emb_mx)
    y_mx_np = np.array(y_mx)

    diff = y_mx_np - y_pt_np
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    flat_pt = y_pt_np.reshape(-1)
    flat_mx = y_mx_np.reshape(-1)
    cosine = float(np.dot(flat_pt, flat_mx) / (np.linalg.norm(flat_pt) * np.linalg.norm(flat_mx) + 1e-12))

    print("Parity metrics (MLX vs PyTorch):")
    print(f"- max_abs_error: {max_abs:.6e}")
    print(f"- mean_abs_error: {mean_abs:.6e}")
    print(f"- rmse: {rmse:.6e}")
    print(f"- cosine_similarity: {cosine:.9f}")


if __name__ == "__main__":
    main()
