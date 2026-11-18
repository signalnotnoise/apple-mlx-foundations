"""Versioned runtime artifact bundle readers and writers."""

from importlib import import_module

__all__ = [
    "ArtifactFamilyManifest",
    "RUNTIME_ARTIFACT_VERSION",
    "RuntimeArtifactBundle",
    "RuntimeArtifactManifest",
    "TensorEntry",
    "save_tensor_family",
]

_LAZY = {
    "ArtifactFamilyManifest": ("mlx_shap_e.runtime_artifacts.manifest", "ArtifactFamilyManifest"),
    "RUNTIME_ARTIFACT_VERSION": ("mlx_shap_e.runtime_artifacts.manifest", "RUNTIME_ARTIFACT_VERSION"),
    "RuntimeArtifactBundle": ("mlx_shap_e.runtime_artifacts.bundle", "RuntimeArtifactBundle"),
    "RuntimeArtifactManifest": ("mlx_shap_e.runtime_artifacts.manifest", "RuntimeArtifactManifest"),
    "TensorEntry": ("mlx_shap_e.runtime_artifacts.manifest", "TensorEntry"),
    "save_tensor_family": ("mlx_shap_e.runtime_artifacts.manifest", "save_tensor_family"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
