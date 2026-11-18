import argparse
from pathlib import Path

import numpy as np
from requests.exceptions import RequestException
from shap_e.models.download import load_model
from shap_e.models.nn.camera import DifferentiableCameraBatch, DifferentiableProjectiveCamera
from shap_e.util.collections import AttrDict
from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from mlx_shap_e.models.rendering.stf_mesh_renderer import decode_latent_mesh_mlx_renderer, decode_latent_mesh_mlx_renderer_artifact


def _decoder_reference_tensor(decoder):
    for param in decoder.parameters():
        return param
    raise ValueError("Decoder has no parameters to infer tensor backend/device.")


def create_pan_cameras(size: int, *, reference_tensor) -> DifferentiableCameraBatch:
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return DifferentiableCameraBatch(
        shape=(1, len(xs)),
        flat_camera=DifferentiableProjectiveCamera(
            origin=reference_tensor.new_tensor(np.stack(origins, axis=0)).float(),
            x=reference_tensor.new_tensor(np.stack(xs, axis=0)).float(),
            y=reference_tensor.new_tensor(np.stack(ys, axis=0)).float(),
            z=reference_tensor.new_tensor(np.stack(zs, axis=0)).float(),
            width=size,
            height=size,
            x_fov=0.7,
            y_fov=0.7,
        ),
    )


def decode_latent_mesh(decoder, latent: np.ndarray):
    ref = _decoder_reference_tensor(decoder)
    latent_t = ref.new_tensor(latent.reshape(1, -1).astype(np.float32)).float()
    decoded = decoder.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, reference_tensor=ref)),
        params=decoder.bottleneck_to_params(latent_t),
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def decode_latent_mesh_mlx_params(decoder, latent: np.ndarray, mlx_decoder: MLXChannelsDecoderParams):
    ref = _decoder_reference_tensor(decoder)
    params = mlx_decoder.bottleneck_to_framework_params(
        latent.reshape(1, -1).astype(np.float32),
        reference_tensor=ref,
    )
    decoded = decoder.renderer.render_views(
        AttrDict(cameras=create_pan_cameras(2, reference_tensor=ref)),
        params=params,
        options=AttrDict(rendering_mode="stf", render_with_direction=False),
    )
    return decoded.raw_meshes[0]


def _load_latent(latent_npz_path: Path) -> np.ndarray:
    data = np.load(latent_npz_path)
    if "final_latent_flat" in data:
        latent = data["final_latent_flat"]
    elif "latent_flat" in data:
        latent = data["latent_flat"]
    elif "final_latent" in data:
        latent = data["final_latent"].reshape(1, -1)
    else:
        raise ValueError(
            "Latent artifact must contain 'final_latent_flat', 'latent_flat', or 'final_latent'."
        )

    if latent.ndim != 2 or latent.shape[0] != 1:
        raise ValueError(f"Expected latent shape [1, D], got {latent.shape}.")

    return latent.astype(np.float32)


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode a sampled text300M latent into mesh files.")
    parser.add_argument(
        "--latent-npz",
        required=True,
        help="Path to trajectory or prompt output .npz artifact.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for mesh outputs (defaults to shap_e/outputs/meshes).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base filename for outputs without extension.",
    )
    parser.add_argument(
        "--decoder-params-backend",
        choices=["torch", "mlx"],
        default="mlx",
        help="Parameter projection backend before renderer (default: torch).",
    )
    parser.add_argument(
        "--renderer-backend",
        choices=["torch", "mlx"],
        default="torch",
        help="Mesh extraction backend (default: torch renderer).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=4096,
        help="Query batch size for MLX renderer SDF grid evaluation.",
    )
    parser.add_argument(
        "--mesh-grid-size",
        type=int,
        default=None,
        help="Override MLX mesh extraction grid size for faster preview meshes.",
    )
    parser.add_argument(
        "--no-mesh-colors",
        action="store_false",
        dest="mesh_colors",
        help="Disable vertex color extraction for mesh outputs.",
    )
    parser.add_argument(
        "--mlx-decoder-artifact",
        default=None,
        help="Path to exported MLX decoder mesh artifact .npz (used by --renderer-backend mlx).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"

    latent_npz_path = Path(args.latent_npz).expanduser().resolve()
    if not latent_npz_path.exists():
        raise FileNotFoundError(f"Latent artifact not found: {latent_npz_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root / "shap_e" / "outputs" / "meshes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.name or latent_npz_path.stem

    print(f"Loading latent from {latent_npz_path}...")
    latent_np = _load_latent(latent_npz_path)
    latent = latent_np[0]

    decoder = None
    artifact_path = (
        _resolve_user_path(args.mlx_decoder_artifact, repo_root=repo_root)
        if args.mlx_decoder_artifact
        else None
    )
    if not (args.renderer_backend == "mlx" and artifact_path is not None):
        print("Loading Shap-E decoder model...")
        try:
            decoder = load_model("decoder", device="cpu", cache_dir=str(cache_dir))
        except RequestException as exc:
            print(f"Failed to fetch decoder artifacts: {exc}")
            print(
                "Tip: ensure network access once to cache vector_decoder files, "
                "or pass --cache-dir pointing to a pre-populated cache."
            )
            raise SystemExit(1)

    print("Decoding latent into mesh...")
    if args.renderer_backend == "mlx":
        if artifact_path is not None:
            mesh = decode_latent_mesh_mlx_renderer_artifact(
                artifact_path,
                latent,
                query_batch_size=args.query_batch_size,
                include_vertex_colors=args.mesh_colors,
                grid_size_override=args.mesh_grid_size,
            ).tri_mesh()
        else:
            mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
            mesh = decode_latent_mesh_mlx_renderer(
                decoder,
                latent,
                mlx_decoder,
                query_batch_size=args.query_batch_size,
                include_vertex_colors=args.mesh_colors,
                grid_size_override=args.mesh_grid_size,
            ).tri_mesh()
    elif args.decoder_params_backend == "mlx":
        mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
        mesh = decode_latent_mesh_mlx_params(decoder, latent, mlx_decoder).tri_mesh()
    else:
        mesh = decode_latent_mesh(decoder, latent).tri_mesh()

    ply_path = output_dir / f"{base_name}.ply"
    obj_path = output_dir / f"{base_name}.obj"
    npz_mesh_path = output_dir / f"{base_name}_mesh.npz"

    with ply_path.open("wb") as f:
        mesh.write_ply(f)
    with obj_path.open("w", encoding="utf-8") as f:
        mesh.write_obj(f)
    mesh.save(str(npz_mesh_path))

    print("Mesh export complete:")
    print(f"- {ply_path}")
    print(f"- {obj_path}")
    print(f"- {npz_mesh_path}")
    print(f"verts={mesh.verts.shape[0]}, faces={mesh.faces.shape[0]}")


if __name__ == "__main__":
    main()
