"""
Shared helpers for MLX ↔ Swift parity dumps.

Reuses the same golden case + pipeline path as check_mlx_runtime_golden.
Full tensors are written for L2 (Swift) comparison; summary heads stay L1.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import mlx.core as mx
import numpy as np

from mlx_shap_e.runtime_core.golden import (
    RuntimeGoldenCase,
    runtime_summary,
    tokenize_prompt,
)


def sanitize_param_key(key: str) -> str:
    """nerstf.mlp.0.weight -> nerstf_mlp_0_weight"""
    return re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")


def to_numpy(array: Any, *, dtype=None) -> np.ndarray:
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def run_mlx_parity_case(
    runtime,
    case: RuntimeGoldenCase,
    *,
    clip_bpe_path: str | Path | None = None,
    progress: bool = False,
    include_vertex_colors: bool = False,
) -> Dict[str, Any]:
    """
    Run the production MLX pipeline for one golden case and return tensors.

    Returns a dict with numpy arrays plus a runtime_summary payload.
    """
    token_ids = tokenize_prompt(
        case.prompt,
        context_length=runtime.clip_artifact.context_length,
        bpe_path=clip_bpe_path,
    )
    embedding = to_numpy(runtime.embed_text_mx([case.prompt]), dtype=np.float32)
    # Match karras_sample_heun_mlx: np.random.default_rng(seed).standard_normal * sigma_max
    # Dump init_x so Swift can use the same noise (MLXRandom ≠ NumPy RNG).
    d_latent = int(runtime.text300m_artifact.d_latent)
    rng = np.random.default_rng(seed=case.seed)
    init_x = (rng.standard_normal((1, d_latent), dtype=np.float32) * np.float32(case.sigma_max)).astype(
        np.float32
    )
    latent_result = runtime.sample_latent_mx(
        embedding,
        seed=case.seed,
        guidance_scale=case.guidance_scale,
        steps=case.steps,
        sigma_min=case.sigma_min,
        sigma_max=case.sigma_max,
        progress=progress,
        init_x=init_x,
    )
    latent = to_numpy(latent_result.latent, dtype=np.float32)
    pred_xstart = to_numpy(latent_result.pred_xstart, dtype=np.float32)
    sigmas = to_numpy(latent_result.sigmas, dtype=np.float32)

    # Decoder params from the same projector the mesh path uses.
    latent_for_params = latent
    if latent_for_params.ndim == 1:
        latent_for_params = latent_for_params[None, :]
    latent_mx = mx.array(latent_for_params.astype(np.float32))
    params_mx = runtime.decoder_params.bottleneck_to_params_mx(latent_mx)

    params_np: Dict[str, np.ndarray] = {
        name: to_numpy(value, dtype=np.float32) for name, value in params_mx.items()
    }

    sdf_fields = runtime.renderer.extract_sdf_fields(
        params_mx,
        query_batch_size=case.query_batch_size,
        grid_size=case.mesh_grid_size,
    )
    sdf_field = to_numpy(sdf_fields[0], dtype=np.float32)

    mesh_result = runtime.decode_latent_to_mesh_mx(
        latent[0] if latent.ndim == 2 else latent,
        query_batch_size=case.query_batch_size,
        include_vertex_colors=include_vertex_colors,
        grid_size=case.mesh_grid_size,
    )
    verts = to_numpy(mesh_result.verts, dtype=np.float32)
    faces = to_numpy(mesh_result.faces, dtype=np.int32)

    summary = runtime_summary(
        case=case,
        token_ids=token_ids,
        embedding=embedding,
        latent=latent,
        pred_xstart=pred_xstart,
        sigmas=sigmas,
        mesh_verts=verts,
        mesh_faces=faces,
        artifact_version=latent_result.artifact_info.artifact_version,
        artifact_family_versions=latent_result.artifact_info.family_versions,
    )

    return {
        "case": case,
        "token_ids": token_ids.astype(np.int32),
        "clip_embedding": embedding.astype(np.float32),
        "init_x": init_x.astype(np.float32),
        "sigmas": sigmas.astype(np.float32),
        "latent": latent.astype(np.float32),
        "pred_xstart": pred_xstart.astype(np.float32),
        "params": params_np,
        "sdf_field": sdf_field,
        "verts": verts,
        "faces": faces.astype(np.int32),
        "summary": summary,
    }


def write_parity_dumps(out_dir: Path, payload: Mapping[str, Any], *, prefix: str = "python") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case: RuntimeGoldenCase = payload["case"]
    (out_dir / "case.json").write_text(
        json.dumps(asdict(case), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    np.save(out_dir / f"{prefix}_token_ids.npy", payload["token_ids"])
    np.save(out_dir / f"{prefix}_clip_embedding.npy", payload["clip_embedding"])
    np.save(out_dir / f"{prefix}_init_x.npy", payload["init_x"])
    np.save(out_dir / f"{prefix}_sigmas.npy", payload["sigmas"])
    np.save(out_dir / f"{prefix}_latent.npy", payload["latent"])
    np.save(out_dir / f"{prefix}_pred_xstart.npy", payload["pred_xstart"])
    np.save(out_dir / f"{prefix}_sdf_field.npy", payload["sdf_field"])
    np.save(out_dir / f"{prefix}_verts.npy", payload["verts"])
    np.save(out_dir / f"{prefix}_faces.npy", payload["faces"])

    archive: Dict[str, np.ndarray] = {
        "token_ids": payload["token_ids"],
        "clip_embedding": payload["clip_embedding"],
        "init_x": payload["init_x"],
        "sigmas": payload["sigmas"],
        "latent": payload["latent"],
        "pred_xstart": payload["pred_xstart"],
        "sdf_field": payload["sdf_field"],
        "verts": payload["verts"],
        "faces": payload["faces"],
    }
    for key, value in payload["params"].items():
        safe = sanitize_param_key(key)
        np.save(out_dir / f"{prefix}_params_{safe}.npy", value)
        archive[f"params_{safe}"] = value

    np.savez_compressed(out_dir / f"{prefix}_golden.npz", **archive)
    (out_dir / f"{prefix}_summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
