from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from shap_e.models.download import load_model
from shap_e.util.collections import AttrDict

from cli.run.decode_text300m_latent_to_mesh import (
    _decoder_reference_tensor,
    _load_latent,
    create_pan_cameras,
)
from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from mlx_shap_e.models.rendering.stf_mesh_renderer import decode_latent_mesh_mlx_renderer


def _bbox(mesh) -> tuple[np.ndarray, np.ndarray]:
    if mesh.verts.size == 0:
        z = np.zeros((3,), dtype=np.float32)
        return z, z
    return mesh.verts.min(axis=0), mesh.verts.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick torch-vs-mlx mesh parity check.")
    parser.add_argument("--latent-npz", required=True, help="Path to latent artifact .npz")
    parser.add_argument("--cache-dir", default=None, help="Path to shap_e cache directory")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Optional override for renderer grid size (smaller is faster).",
    )
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument(
        "--strict-non-empty",
        action="store_true",
        help="Exit with code 2 if torch/MLX non-empty agreement fails.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"

    latent_np = _load_latent(Path(args.latent_npz).expanduser().resolve())[0]

    decoder = load_model("decoder", device="cpu", cache_dir=str(cache_dir))
    if args.grid_size is not None:
        decoder.renderer.grid_size = int(args.grid_size)

    ref = _decoder_reference_tensor(decoder)
    latent_t = ref.new_tensor(latent_np.reshape(1, -1)).float()
    decoded_torch = decoder.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, reference_tensor=ref)),
        params=decoder.bottleneck_to_params(latent_t),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    torch_non_empty = bool(decoded_torch.mesh_mask[0].item())
    mesh_torch = decoded_torch.raw_meshes[0].tri_mesh()
    if not torch_non_empty:
        mesh_torch.verts = np.zeros((0, 3), dtype=np.float32)
        mesh_torch.faces = np.zeros((0, 3), dtype=np.int64)

    mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
    mesh_mlx = decode_latent_mesh_mlx_renderer(
        decoder,
        latent_np,
        mlx_decoder,
        query_batch_size=args.query_batch_size,
    ).tri_mesh()
    mlx_non_empty = mesh_mlx.faces.shape[0] > 0

    t_min, t_max = _bbox(mesh_torch)
    m_min, m_max = _bbox(mesh_mlx)

    print("Torch mesh:")
    print(f"- verts={mesh_torch.verts.shape[0]} faces={mesh_torch.faces.shape[0]}")
    print(f"- non_empty={torch_non_empty}")
    print(f"- bbox_min={t_min}")
    print(f"- bbox_max={t_max}")

    print("MLX mesh:")
    print(f"- verts={mesh_mlx.verts.shape[0]} faces={mesh_mlx.faces.shape[0]}")
    print(f"- non_empty={mlx_non_empty}")
    print(f"- bbox_min={m_min}")
    print(f"- bbox_max={m_max}")

    if args.strict_non_empty and torch_non_empty != mlx_non_empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
