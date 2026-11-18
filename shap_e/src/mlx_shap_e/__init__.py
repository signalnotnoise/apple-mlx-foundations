"""MLX Shap-E package."""

from importlib import import_module

__all__ = [
    "Text300MTransformer",
    "timestep_embedding",
    "karras_sample_heun_mlx",
    "split_vector_forward_mlx",
    "MlxClipTextEmbedder",
    "load_clip_embedding_artifact",
    "MLXChannelsDecoderParams",
]

_LAZY = {
    "Text300MTransformer": ("mlx_shap_e.models.text300m", "Text300MTransformer"),
    "timestep_embedding": ("mlx_shap_e.models.text300m", "timestep_embedding"),
    "karras_sample_heun_mlx": ("mlx_shap_e.diffusion.karras_sampler", "karras_sample_heun_mlx"),
    "split_vector_forward_mlx": ("mlx_shap_e.diffusion.step", "split_vector_forward_mlx"),
    "MlxClipTextEmbedder": ("mlx_shap_e.embeddings.clip_text_embedder", "MlxClipTextEmbedder"),
    "load_clip_embedding_artifact": (
        "mlx_shap_e.embeddings.clip_embedding_artifact",
        "load_clip_embedding_artifact",
    ),
    "MLXChannelsDecoderParams": ("mlx_shap_e.models.decoder_params", "MLXChannelsDecoderParams"),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
