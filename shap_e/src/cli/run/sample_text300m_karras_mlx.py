from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import torch
from requests.exceptions import RequestException
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import (
    DEFAULT_KARRAS_S_CHURN,
    DEFAULT_KARRAS_SIGMA_MAX,
    DEFAULT_KARRAS_SIGMA_MIN,
)
from shap_e.models.download import load_config, load_model

# Allow direct execution: python ./cli/run/sample_text300m_karras_mlx.py
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlx_shap_e.models.embeddings.clip_embedding_artifact import load_clip_embedding_artifact
from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from cli.run.decode_text300m_latent_to_mesh import decode_latent_mesh, decode_latent_mesh_mlx_params
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.rendering.stf_mesh_renderer import decode_latent_mesh_mlx_renderer, decode_latent_mesh_mlx_renderer_artifact
from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from mlx_shap_e.models.diffusion.karras_sampler import karras_sample_heun_mlx
from mlx_shap_e.models.text300m import Text300MTransformer


def _slugify(text: str, max_len: int = 40) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "prompt")[:max_len]


def _resolve_user_path(path_str: str, *, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text300M latent with MLX Karras Heun loop.")
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Classifier-free guidance scale.")
    parser.add_argument("--karras-steps", type=int, default=64, help="Number of Karras sampling steps.")
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_KARRAS_SIGMA_MIN, help="Minimum Karras sigma.")
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_KARRAS_SIGMA_MAX, help="Maximum Karras sigma.")
    parser.add_argument("--s-churn", type=float, default=DEFAULT_KARRAS_S_CHURN, help="Karras churn parameter.")
    parser.add_argument(
        "--integrator-math",
        choices=["torch", "mlx"],
        default="mlx",
        help="Karras schedule/integrator math backend (default: torch for parity lock).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output latent .npz path (defaults to shap_e/outputs/mlx_karras_<prompt>.npz).",
    )
    parser.add_argument(
        "--decode-mesh",
        action="store_true",
        help="Also decode sampled latent and export .obj/.ply mesh files.",
    )
    parser.add_argument(
        "--mesh-output-dir",
        default=None,
        help="Directory for decoded meshes (defaults to <latent_output_dir>/meshes).",
    )
    parser.add_argument(
        "--decoder-params-backend",
        choices=["torch", "mlx"],
        default="mlx",
        help="Parameter projection backend before renderer when --decode-mesh is used.",
    )
    parser.add_argument(
        "--renderer-backend",
        choices=["torch", "mlx"],
        default="mlx",
        help="Mesh extraction backend when --decode-mesh is used.",
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
    total_t0 = time.perf_counter()

    project_root = Path(__file__).resolve().parents[4]
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else project_root / "shap_e_model_cache"
    )

    outputs_dir = project_root / "shap_e" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else outputs_dir / f"mlx_karras_{_slugify(args.prompt)}.npz"
    )
    clip_text_artifact_path = (
        _resolve_user_path(args.mlx_clip_text_artifact, repo_root=project_root)
        if args.mlx_clip_text_artifact
        else project_root / "shap_e" / "outputs" / "artifacts" / "mlx_clip_text_vit_l14_v1.npz"
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
        emb_np = embedder.embed_text_mx([args.prompt])
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

    print("Sampling latent with MLX Karras Heun loop...")
    sample_t0 = time.perf_counter()
    out = karras_sample_heun_mlx(
        diffusion=diffusion,
        mlx_model=mlx_model,
        embeddings=emb_np,
        shape=(1, pt_model.d_latent),
        steps=args.karras_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        guidance_scale=args.guidance_scale,
        clip_denoised=True,
        seed=args.seed,
        n_ctx=pt_model.n_ctx,
        progress=True,
        strict_torch_math=(args.integrator_math == "torch"),
    )
    sample_seconds = time.perf_counter() - sample_t0

    latent = out["x"].astype(np.float32)
    np.savez_compressed(
        output_path,
        prompt=np.array([args.prompt]),
        seed=np.array([args.seed], dtype=np.int32),
        guidance_scale=np.array([args.guidance_scale], dtype=np.float32),
        karras_steps=np.array([args.karras_steps], dtype=np.int32),
        sigma_min=np.array([args.sigma_min], dtype=np.float32),
        sigma_max=np.array([args.sigma_max], dtype=np.float32),
        integrator_math=np.array([args.integrator_math]),
        latent_flat=latent,
        final_latent_flat=latent,
        pred_xstart_flat=out["pred_xstart"],
        sigmas=out["sigmas"],
        clip_embedding=emb_np,
    )

    print(f"Saved MLX Karras latent artifact: {output_path}")
    print(f"latent shape: {latent.shape}")

    if args.decode_mesh:
        mesh_output_dir = (
            Path(args.mesh_output_dir).expanduser().resolve()
            if args.mesh_output_dir
            else output_path.parent / "meshes"
        )
        mesh_output_dir.mkdir(parents=True, exist_ok=True)
        base_name = output_path.stem
        ply_path = mesh_output_dir / f"{base_name}.ply"
        obj_path = mesh_output_dir / f"{base_name}.obj"
        mesh_npz_path = mesh_output_dir / f"{base_name}_mesh.npz"

        decoder = None
        artifact_path = (
            _resolve_user_path(args.mlx_decoder_artifact, repo_root=project_root)
            if args.mlx_decoder_artifact
            else None
        )
        if not (args.renderer_backend == "mlx" and artifact_path is not None):
            print("Loading decoder model...")
            try:
                decoder = load_model("decoder", device=torch.device("cpu"), cache_dir=str(cache_dir))
            except RequestException as exc:
                print(f"Failed to fetch decoder artifacts: {exc}")
                print(
                    "Tip: ensure network access once to cache vector_decoder files, "
                    "or pass --cache-dir pointing to a pre-populated cache."
                )
                raise SystemExit(1)

        print("Decoding mesh...")
        if args.renderer_backend == "mlx":
            if artifact_path is not None:
                mesh = decode_latent_mesh_mlx_renderer_artifact(
                    artifact_path,
                    latent[0],
                    query_batch_size=args.query_batch_size,
                    include_vertex_colors=args.mesh_colors,
                    grid_size_override=args.mesh_grid_size,
                ).tri_mesh()
            else:
                mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
                mesh = decode_latent_mesh_mlx_renderer(
                    decoder,
                    latent[0],
                    mlx_decoder,
                    query_batch_size=args.query_batch_size,
                    include_vertex_colors=args.mesh_colors,
                    grid_size_override=args.mesh_grid_size,
                ).tri_mesh()
        elif args.decoder_params_backend == "mlx":
            mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
            mesh = decode_latent_mesh_mlx_params(decoder, latent[0], mlx_decoder).tri_mesh()
        else:
            mesh = decode_latent_mesh(decoder, latent[0]).tri_mesh()
        with ply_path.open("wb") as f:
            mesh.write_ply(f)
        with obj_path.open("w", encoding="utf-8") as f:
            mesh.write_obj(f)
        mesh.save(str(mesh_npz_path))

        print("Mesh export complete:")
        print(f"- {ply_path}")
        print(f"- {obj_path}")
        print(f"- {mesh_npz_path}")

    total_seconds = time.perf_counter() - total_t0
    print(
        f"Timing: integrator={args.integrator_math} "
        f"sample={sample_seconds:.3f}s total={total_seconds:.3f}s"
    )


if __name__ == "__main__":
    main()
