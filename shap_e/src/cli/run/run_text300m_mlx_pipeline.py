from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from shap_e.diffusion.sample import (
    DEFAULT_KARRAS_S_CHURN,
    DEFAULT_KARRAS_SIGMA_MAX,
    DEFAULT_KARRAS_SIGMA_MIN,
)
from shap_e.rendering.mesh import TriMesh

from mlx_shap_e.offline_tools.runtime_artifact_exports import prepare_runtime_artifacts
from mlx_shap_e.runtime_artifacts.bundle import RuntimeArtifactBundle
from mlx_shap_e.runtime_core.pipeline import MLXShapERuntime


def _slugify(text: str, max_len: int = 48) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "prompt")[:max_len]


def _required_families_present(bundle_dir: Path) -> bool:
    try:
        bundle = RuntimeArtifactBundle(bundle_dir)
    except FileNotFoundError:
        return False
    required = {"clip_text", "text300m", "diffusion", "decoder_params", "renderer"}
    return required.issubset(bundle.manifest.families.keys())


def _mesh_to_trimesh(mesh_result) -> TriMesh:
    vertex_channels = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in (mesh_result.vertex_channels or {}).items()
    }
    return TriMesh(
        verts=np.asarray(mesh_result.verts, dtype=np.float32),
        faces=np.asarray(mesh_result.faces, dtype=np.int64),
        vertex_channels=vertex_channels,
    )


def _load_prompts(*, prompts: list[str] | None, prompts_file: str | None) -> list[str]:
    out: list[str] = []
    for prompt in prompts or []:
        prompt = prompt.strip()
        if prompt:
            out.append(prompt)
    if prompts_file:
        prompts_path = Path(prompts_file).expanduser().resolve()
        if not prompts_path.exists():
            raise FileNotFoundError(
                f"Prompts file not found: {prompts_path}. "
                "Pass a real path to --prompts-file, or repeat --prompt for each prompt."
            )
        for line in prompts_path.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if prompt:
                out.append(prompt)
    if not out:
        raise ValueError("Provide at least one --prompt or --prompts-file with one prompt per line.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline harness for the MLX-only text300M runtime bundle."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Text prompt. Repeat this flag to run a batched prompt decode.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Optional text file with one prompt per line for batched runs.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Classifier-free guidance scale.")
    parser.add_argument("--karras-steps", type=int, default=64, help="Number of Karras steps.")
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_KARRAS_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_KARRAS_SIGMA_MAX)
    parser.add_argument("--s-churn", type=float, default=DEFAULT_KARRAS_S_CHURN)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (default: repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--runtime-artifacts-dir",
        default=None,
        help="Directory containing manifest.json plus raw tensor blobs for the runtime bundle.",
    )
    parser.add_argument(
        "--skip-artifact-preparation",
        action="store_true",
        help="Do not run offline exporters when the runtime bundle is missing or incomplete.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: shap_e/outputs/pipeline).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Base name for output files (default: mlx_pipeline_<slug(prompt)> or mlx_pipeline_batch_<slug(first_prompt)>).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=4096,
        help="Query batch size for SDF grid evaluation.",
    )
    parser.add_argument(
        "--mesh-grid-size",
        type=int,
        default=None,
        help="Override mesh extraction grid size for faster preview meshes.",
    )
    parser.add_argument(
        "--no-mesh-colors",
        action="store_false",
        dest="mesh_colors",
        help="Disable vertex color extraction for mesh outputs.",
    )
    parser.add_argument(
        "--clip-bpe-path",
        default=None,
        help="Path to CLIP tokenizer merges file bpe_simple_vocab_16e6.txt.gz.",
    )
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[4]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root / "shap_e" / "outputs" / "pipeline"
    )
    runtime_artifacts_dir = (
        Path(args.runtime_artifacts_dir).expanduser().resolve()
        if args.runtime_artifacts_dir
        else repo_root / "shap_e" / "outputs" / "artifacts" / "runtime_bundle_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = _load_prompts(prompts=args.prompt, prompts_file=args.prompts_file)
    is_batch = len(prompts) > 1

    if not _required_families_present(runtime_artifacts_dir):
        if args.skip_artifact_preparation:
            raise FileNotFoundError(
                f"Runtime artifact bundle is missing or incomplete at {runtime_artifacts_dir}. "
                "Run the offline exporters first or remove --skip-artifact-preparation."
            )
        print(f"Preparing runtime artifact bundle at {runtime_artifacts_dir}...")
        prepare_runtime_artifacts(bundle_dir=runtime_artifacts_dir, cache_dir=cache_dir)

    runtime = MLXShapERuntime(runtime_artifacts_dir, tokenizer_bpe_path=args.clip_bpe_path)

    base = args.name or (
        f"mlx_pipeline_batch_{_slugify(prompts[0])}" if is_batch else f"mlx_pipeline_{_slugify(prompts[0])}"
    )
    latent_npz_path = output_dir / f"{base}_latent.npz"
    metadata_json_path = output_dir / f"{base}_metadata.json"

    print(f"Embedding {len(prompts)} prompt(s) with MLX CLIP runtime...")
    token_ids = runtime.embedder.tokenizer.tokenize(
        prompts,
        context_length=runtime.clip_artifact.context_length,
    )
    embed_t0 = time.perf_counter()
    embedding = runtime.embed_text_mx(prompts)
    embed_seconds = time.perf_counter() - embed_t0

    print("Sampling latent with MLX diffusion runtime...")
    latent_result = runtime.sample_latent_mx(
        embedding,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        steps=args.karras_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        progress=True,
    )

    print(f"Decoding {len(prompts)} latent(s) to mesh with runtime renderer...")
    mesh_batch = runtime.decode_latents_to_meshes_mx(
        latent_result.latent,
        query_batch_size=args.query_batch_size,
        include_vertex_colors=args.mesh_colors,
        grid_size=args.mesh_grid_size,
    )

    latent_np = np.asarray(latent_result.latent, dtype=np.float32)
    pred_xstart_np = np.asarray(latent_result.pred_xstart, dtype=np.float32)
    embedding_np = np.asarray(embedding, dtype=np.float32)
    sigmas_np = np.asarray(latent_result.sigmas, dtype=np.float32)

    np.savez_compressed(
        latent_npz_path,
        prompt=np.array(prompts),
        token_ids=token_ids.astype(np.int32),
        seed=np.array([args.seed], dtype=np.int32),
        guidance_scale=np.array([args.guidance_scale], dtype=np.float32),
        karras_steps=np.array([args.karras_steps], dtype=np.int32),
        sigma_min=np.array([args.sigma_min], dtype=np.float32),
        sigma_max=np.array([args.sigma_max], dtype=np.float32),
        s_churn=np.array([args.s_churn], dtype=np.float32),
        latent_flat=latent_np,
        final_latent_flat=latent_np,
        pred_xstart_flat=pred_xstart_np,
        sigmas=sigmas_np,
        clip_embedding=embedding_np,
        runtime_artifacts_dir=np.array([str(runtime_artifacts_dir)]),
        artifact_manifest=np.array([latent_result.artifact_info.manifest_path]),
    )

    metadata = {
        "prompts": prompts,
        "runtime_artifacts_dir": str(runtime_artifacts_dir),
        "artifact_manifest": latent_result.artifact_info.manifest_path,
        "artifact_version": latent_result.artifact_info.artifact_version,
        "artifact_family_versions": latent_result.artifact_info.family_versions,
        "token_ids": token_ids.astype(int).tolist(),
        "seed": int(args.seed),
        "guidance_scale": float(args.guidance_scale),
        "karras_steps": int(args.karras_steps),
        "sigma_min": float(args.sigma_min),
        "sigma_max": float(args.sigma_max),
        "s_churn": float(args.s_churn),
        "query_batch_size": int(args.query_batch_size),
        "mesh_grid_size_override": int(args.mesh_grid_size) if args.mesh_grid_size is not None else None,
        "latent_npz": str(latent_npz_path),
        "timings": {
            "embed_seconds": embed_seconds,
            **latent_result.timings,
            **mesh_batch.timings,
            "total_seconds": time.perf_counter() - total_t0,
        },
    }

    if not is_batch:
        mesh_result = mesh_batch.meshes[0]
        mesh_npz_path = output_dir / f"{base}_mesh.npz"
        mesh_ply_path = output_dir / f"{base}.ply"
        mesh_obj_path = output_dir / f"{base}.obj"
        tri_mesh = _mesh_to_trimesh(mesh_result)
        with mesh_ply_path.open("wb") as handle:
            tri_mesh.write_ply(handle)
        with mesh_obj_path.open("w", encoding="utf-8") as handle:
            tri_mesh.write_obj(handle)
        tri_mesh.save(str(mesh_npz_path))

        metadata.update(
            {
                "prompt": prompts[0],
                "mesh_npz": str(mesh_npz_path),
                "mesh_ply": str(mesh_ply_path),
                "mesh_obj": str(mesh_obj_path),
                "mesh_vert_count": int(np.asarray(mesh_result.verts).shape[0]),
                "mesh_face_count": int(np.asarray(mesh_result.faces).shape[0]),
            }
        )
    else:
        mesh_output_dir = output_dir / f"{base}_meshes"
        mesh_output_dir.mkdir(parents=True, exist_ok=True)
        mesh_outputs = []
        for idx, (prompt, mesh_result) in enumerate(zip(prompts, mesh_batch.meshes)):
            sample_base = f"{idx:02d}_{_slugify(prompt, max_len=32)}"
            mesh_npz_path = mesh_output_dir / f"{sample_base}_mesh.npz"
            mesh_ply_path = mesh_output_dir / f"{sample_base}.ply"
            mesh_obj_path = mesh_output_dir / f"{sample_base}.obj"
            tri_mesh = _mesh_to_trimesh(mesh_result)
            with mesh_ply_path.open("wb") as handle:
                tri_mesh.write_ply(handle)
            with mesh_obj_path.open("w", encoding="utf-8") as handle:
                tri_mesh.write_obj(handle)
            tri_mesh.save(str(mesh_npz_path))
            mesh_outputs.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "mesh_npz": str(mesh_npz_path),
                    "mesh_ply": str(mesh_ply_path),
                    "mesh_obj": str(mesh_obj_path),
                    "mesh_vert_count": int(np.asarray(mesh_result.verts).shape[0]),
                    "mesh_face_count": int(np.asarray(mesh_result.faces).shape[0]),
                }
            )
        metadata["mesh_output_dir"] = str(mesh_output_dir)
        metadata["mesh_outputs"] = mesh_outputs

    metadata_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Pipeline complete:")
    print(f"- latent: {latent_npz_path}")
    if not is_batch:
        print(f"- mesh npz: {mesh_npz_path}")
        print(f"- mesh ply: {mesh_ply_path}")
        print(f"- mesh obj: {mesh_obj_path}")
        print(f"verts={tri_mesh.verts.shape[0]}, faces={tri_mesh.faces.shape[0]}")
    else:
        print(f"- meshes: {mesh_output_dir}")
        for sample in metadata["mesh_outputs"]:
            print(
                f"  [{sample['index']}] verts={sample['mesh_vert_count']} "
                f"faces={sample['mesh_face_count']} "
                f"{sample['mesh_ply']}"
            )
    print(f"- metadata: {metadata_json_path}")
    print(
        "Timing: "
        f"embed={embed_seconds:.3f}s "
        f"sample={latent_result.timings.get('sample_seconds', 0.0):.3f}s "
        f"decode={mesh_batch.timings.get('total_seconds', 0.0):.3f}s "
        f"total={time.perf_counter() - total_t0:.3f}s"
    )


if __name__ == "__main__":
    main()
