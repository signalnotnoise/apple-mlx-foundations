from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_shap_e.models.rendering.stf_mesh_renderer import MLXMeshBatchResult, MLXMeshResult, MLXSTFMeshRenderer
from mlx_shap_e.runtime_core.pipeline import MLXShapERuntime
from mlx_shap_e.runtime_core.types import ArtifactVersionInfo, MeshBatchResult, MeshResult


def _dummy_renderer() -> MLXSTFMeshRenderer:
    return MLXSTFMeshRenderer(
        grid_size=4,
        bbox_min=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        bbox_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        static_weights={
            4: np.zeros((8, 67), dtype=np.float32),
            5: np.zeros((8, 8), dtype=np.float32),
            6: np.zeros((6, 8), dtype=np.float32),
        },
        static_biases={
            0: np.zeros((4,), dtype=np.float32),
            1: np.zeros((4,), dtype=np.float32),
            2: np.zeros((8,), dtype=np.float32),
            3: np.zeros((16,), dtype=np.float32),
            4: np.zeros((8,), dtype=np.float32),
            5: np.zeros((8,), dtype=np.float32),
            6: np.zeros((6,), dtype=np.float32),
        },
    )


def test_extract_meshes_keeps_batch_members_separate(monkeypatch):
    renderer = _dummy_renderer()

    def fake_signed_distance(_positions, dynamic_params):
        batch_size = int(dynamic_params["dummy"].shape[0])
        num_points = int(_positions.shape[0])
        neg = mx.full((1, num_points, 1), -1.0, dtype=mx.float32)
        pos = mx.full((batch_size - 1, num_points, 1), 1.0, dtype=mx.float32)
        return mx.concatenate([neg, pos], axis=0)

    monkeypatch.setattr(renderer, "signed_distance_mx", fake_signed_distance)

    mesh_batch = renderer.extract_meshes(
        {"dummy": mx.zeros((2, 1), dtype=mx.float32)},
        query_batch_size=8,
        include_vertex_colors=False,
        grid_size=4,
    )

    assert isinstance(mesh_batch, MLXMeshBatchResult)
    assert len(mesh_batch) == 2
    assert int(mesh_batch.meshes[0].verts.shape[0]) == 0
    assert int(mesh_batch.meshes[0].faces.shape[0]) == 0
    assert int(mesh_batch.meshes[1].verts.shape[0]) > 0
    assert int(mesh_batch.meshes[1].faces.shape[0]) > 0


def test_runtime_decode_latent_to_mesh_returns_batch_result_for_batched_latents():
    runtime = MLXShapERuntime.__new__(MLXShapERuntime)
    runtime.artifact_info = ArtifactVersionInfo(
        artifact_version=1,
        manifest_path="test-manifest.json",
        family_versions={"renderer": 1},
    )
    runtime.decoder_params = SimpleNamespace(
        bottleneck_to_params_mx=lambda latent_mx: {"latent": latent_mx}
    )

    def fake_extract_meshes(params_mx, **_kwargs):
        batch_size = int(params_mx["latent"].shape[0])
        meshes = []
        for idx in range(batch_size):
            verts = mx.full((idx + 1, 3), float(idx), dtype=mx.float32)
            faces = mx.zeros((0, 3), dtype=mx.int32)
            meshes.append(MLXMeshResult(verts=verts, faces=faces))
        return MLXMeshBatchResult(meshes=tuple(meshes))

    runtime.renderer = SimpleNamespace(
        bbox_min=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        bbox_max=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        extract_meshes=fake_extract_meshes,
    )

    batch = runtime.decode_latents_to_meshes_mx(mx.zeros((2, 4), dtype=mx.float32))
    single = runtime.decode_latent_to_mesh_mx(mx.zeros((4,), dtype=mx.float32))

    assert isinstance(batch, MeshBatchResult)
    assert len(batch) == 2
    assert int(batch.meshes[0].verts.shape[0]) == 1
    assert int(batch.meshes[1].verts.shape[0]) == 2

    assert isinstance(single, MeshResult)
    assert int(single.verts.shape[0]) == 1
    assert "total_seconds" in single.timings
