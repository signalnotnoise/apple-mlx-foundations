from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mlx_shap_e.runtime_core.golden import (
    RuntimeGoldenCase,
    compare_runtime_summary,
    load_golden,
    runtime_summary,
    tokenize_prompt,
)


@pytest.mark.skipif(
    os.environ.get("MLX_SHAP_E_RUN_GOLDENS") != "1",
    reason="Set MLX_SHAP_E_RUN_GOLDENS=1 to run MLX integration goldens.",
)
def test_runtime_golden_a_red_chair_steps2():
    repo_root = Path(__file__).resolve().parents[1]
    golden_path = repo_root / "tests" / "fixtures" / "runtime_golden_a_red_chair_steps2.json"
    bundle_dir = repo_root / "shap_e" / "outputs" / "artifacts" / "runtime_bundle_v1"
    if not bundle_dir.exists():
        pytest.skip(f"Runtime bundle not found: {bundle_dir}")

    from mlx_shap_e.runtime_core.pipeline import MLXShapERuntime

    expected = load_golden(golden_path)
    case = RuntimeGoldenCase(**expected["case"])
    runtime = MLXShapERuntime(bundle_dir)
    token_ids = tokenize_prompt(case.prompt, context_length=runtime.clip_artifact.context_length)
    embedding = np.asarray(runtime.embed_text_mx([case.prompt]), dtype=np.float32)
    latent_result = runtime.sample_latent_mx(
        embedding,
        seed=case.seed,
        guidance_scale=case.guidance_scale,
        steps=case.steps,
        sigma_min=case.sigma_min,
        sigma_max=case.sigma_max,
    )
    mesh_result = runtime.decode_latent_to_mesh_mx(
        latent_result.latent[0],
        query_batch_size=case.query_batch_size,
        include_vertex_colors=True,
        grid_size=case.mesh_grid_size,
    )

    actual = runtime_summary(
        case=case,
        token_ids=token_ids,
        embedding=embedding,
        latent=np.asarray(latent_result.latent, dtype=np.float32),
        pred_xstart=np.asarray(latent_result.pred_xstart, dtype=np.float32),
        sigmas=np.asarray(latent_result.sigmas, dtype=np.float32),
        mesh_verts=np.asarray(mesh_result.verts, dtype=np.float32),
        mesh_faces=np.asarray(mesh_result.faces, dtype=np.int32),
        artifact_version=latent_result.artifact_info.artifact_version,
        artifact_family_versions=latent_result.artifact_info.family_versions,
    )
    compare_runtime_summary(actual, expected, atol=1e-4)
