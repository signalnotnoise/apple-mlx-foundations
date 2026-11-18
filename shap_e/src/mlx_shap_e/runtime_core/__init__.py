"""Production MLX runtime surface for Shap-E."""

from importlib import import_module

__all__ = [
    "ArtifactVersionInfo",
    "BoundingBox",
    "LatentResult",
    "MeshBatchResult",
    "MeshResult",
    "MLXShapERuntime",
    "RuntimeDiffusionConfig",
]

_LAZY = {
    "ArtifactVersionInfo": ("mlx_shap_e.runtime_core.types", "ArtifactVersionInfo"),
    "BoundingBox": ("mlx_shap_e.runtime_core.types", "BoundingBox"),
    "LatentResult": ("mlx_shap_e.runtime_core.types", "LatentResult"),
    "MeshBatchResult": ("mlx_shap_e.runtime_core.types", "MeshBatchResult"),
    "MeshResult": ("mlx_shap_e.runtime_core.types", "MeshResult"),
    "MLXShapERuntime": ("mlx_shap_e.runtime_core.pipeline", "MLXShapERuntime"),
    "RuntimeDiffusionConfig": ("mlx_shap_e.runtime_core.diffusion", "RuntimeDiffusionConfig"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
