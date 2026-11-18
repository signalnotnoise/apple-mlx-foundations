from __future__ import annotations

import numpy as np
import torch
from shap_e.rendering.mc import marching_cubes as torch_marching_cubes

from mlx_shap_e.models.rendering.marching_cubes import marching_cubes
from mlx_shap_e.models.rendering.stf_mesh_renderer import MLXMeshResult


def test_mlx_marching_cubes_matches_shap_e_torch():
    grid_size = 18
    coords = np.linspace(-1.2, 1.2, grid_size, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    field = (0.55 - np.sqrt(x * x + y * y + z * z)).astype(np.float32)

    min_point = np.array([-0.8, -0.6, -0.4], dtype=np.float32)
    size = np.array([1.6, 1.4, 1.2], dtype=np.float32)

    verts_np, faces_np = marching_cubes(field, min_point, size)
    verts_np = np.asarray(verts_np, dtype=np.float32)
    faces_np = np.asarray(faces_np, dtype=np.int64)
    mesh_torch = torch_marching_cubes(
        torch.from_numpy(field),
        torch.from_numpy(min_point),
        torch.from_numpy(size),
    )

    verts_torch = mesh_torch.verts.detach().cpu().numpy().astype(np.float32)
    faces_torch = mesh_torch.faces.detach().cpu().numpy().astype(np.int64)

    assert verts_np.shape == verts_torch.shape
    assert faces_np.shape == faces_torch.shape
    assert np.allclose(verts_np, verts_torch, atol=1e-6, rtol=0.0)
    assert np.array_equal(faces_np, faces_torch)


def test_mlx_mesh_result_tri_mesh_converts_public_arrays():
    field = np.array(
        [
            [[-1.0, -1.0], [-1.0, -1.0]],
            [[-1.0, -1.0], [-1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    verts, faces = marching_cubes(
        field,
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    mesh_result = MLXMeshResult(
        verts=verts,
        faces=faces,
    )

    tri_mesh = mesh_result.tri_mesh()

    assert isinstance(tri_mesh.verts, np.ndarray)
    assert isinstance(tri_mesh.faces, np.ndarray)
