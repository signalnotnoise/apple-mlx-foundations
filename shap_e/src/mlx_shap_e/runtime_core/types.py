from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import mlx.core as mx
import numpy as np
from shap_e.rendering.mesh import TriMesh


@dataclass(frozen=True)
class ArtifactVersionInfo:
    artifact_version: int
    manifest_path: str
    family_versions: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundingBox:
    min_corner: mx.array
    max_corner: mx.array


@dataclass
class LatentResult:
    latent: mx.array
    pred_xstart: mx.array
    sigmas: mx.array
    seed: int
    guidance_scale: float
    steps: int
    timings: Dict[str, float]
    artifact_info: ArtifactVersionInfo


@dataclass
class MeshResult:
    verts: mx.array
    faces: mx.array
    vertex_channels: Dict[str, mx.array] | None
    bbox: BoundingBox
    timings: Dict[str, float]
    artifact_info: ArtifactVersionInfo

    def tri_mesh(self) -> TriMesh:
        vertex_channels = {
            name: np.asarray(values, dtype=np.float32)
            for name, values in (self.vertex_channels or {}).items()
        }
        return TriMesh(
            verts=np.asarray(self.verts, dtype=np.float32),
            faces=np.asarray(self.faces, dtype=np.int64),
            vertex_channels=vertex_channels,
        )


@dataclass
class MeshBatchResult:
    meshes: tuple[MeshResult, ...]
    bbox: BoundingBox
    timings: Dict[str, float]
    artifact_info: ArtifactVersionInfo

    def __len__(self) -> int:
        return len(self.meshes)

    def __iter__(self):
        return iter(self.meshes)

    def tri_meshes(self) -> tuple[TriMesh, ...]:
        return tuple(mesh.tri_mesh() for mesh in self.meshes)
