from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_config, load_model

from mlx_shap_e.models.conversions.convert_text300m_to_mlx import map_text300m_weights
from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from mlx_shap_e.models.rendering.stf_mesh_renderer import MLXSTFMeshRenderer
from mlx_shap_e.runtime_artifacts.manifest import save_tensor_family
from mlx_shap_e.runtime_core.diffusion import RuntimeDiffusionConfig
from mlx_shap_e.utils.cache_recovery import with_corrupt_cache_retry


def _load_model_with_retry(name: str, cache_dir: Path):
    def _load():
        return load_model(name, device=torch.device("cpu"), cache_dir=str(cache_dir))

    def _on_retry(path: Path) -> None:
        print(f"Detected corrupted cache file; deleting and retrying: {path}")

    return with_corrupt_cache_retry(_load, on_retry=_on_retry, max_attempts=2)


def export_clip_text_runtime_family(
    *,
    bundle_dir: Path,
    cache_dir: Path,
    clip_name: str = "ViT-L/14",
) -> None:
    import clip

    model, _ = clip.load(clip_name, device=torch.device("cpu"), download_root=str(cache_dir))
    state = model.state_dict()
    n_layers = len([k for k in state.keys() if k.startswith("transformer.resblocks.") and k.endswith(".ln_1.weight")])
    metadata = {
        "clip_name": clip_name,
        "context_length": int(model.context_length),
        "transformer_width": int(model.transformer.width),
        "transformer_heads": int(model.transformer.width // 64),
        "transformer_layers": int(n_layers),
        "tokenizer": {
            "type": "clip_bpe",
            "context_length": int(model.context_length),
        },
    }
    tensors = {
        "token_embedding.weight": state["token_embedding.weight"].detach().cpu().numpy().astype(np.float32),
        "positional_embedding": state["positional_embedding"].detach().cpu().numpy().astype(np.float32),
        "ln_final.weight": state["ln_final.weight"].detach().cpu().numpy().astype(np.float32),
        "ln_final.bias": state["ln_final.bias"].detach().cpu().numpy().astype(np.float32),
        "text_projection": state["text_projection"].detach().cpu().numpy().astype(np.float32),
    }
    for i in range(n_layers):
        prefix = f"transformer.resblocks.{i}"
        for name in (
            "ln_1.weight",
            "ln_1.bias",
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ):
            key = f"{prefix}.{name}"
            tensors[key] = state[key].detach().cpu().numpy().astype(np.float32)
    save_tensor_family(bundle_dir, family="clip_text", metadata=metadata, tensors=tensors)


def export_clip_vision_runtime_family(
    *,
    bundle_dir: Path,
    cache_dir: Path,
    clip_name: str = "ViT-L/14",
) -> None:
    import clip

    model, _ = clip.load(clip_name, device=torch.device("cpu"), download_root=str(cache_dir))
    state = model.visual.state_dict()
    n_layers = len([k for k in state.keys() if k.startswith("transformer.resblocks.") and k.endswith(".ln_1.weight")])
    metadata = {
        "clip_name": clip_name,
        "input_resolution": int(model.visual.input_resolution),
        "patch_size": int(model.visual.conv1.weight.shape[2]),
        "transformer_width": int(model.visual.transformer.width),
        "transformer_heads": int(model.visual.transformer.width // 64),
        "transformer_layers": int(n_layers),
        "output_dim": int(model.visual.output_dim),
    }
    tensors = {
        "class_embedding": state["class_embedding"].detach().cpu().numpy().astype(np.float32),
        "positional_embedding": state["positional_embedding"].detach().cpu().numpy().astype(np.float32),
        "conv1.weight": state["conv1.weight"].detach().cpu().numpy().astype(np.float32),
        "ln_pre.weight": state["ln_pre.weight"].detach().cpu().numpy().astype(np.float32),
        "ln_pre.bias": state["ln_pre.bias"].detach().cpu().numpy().astype(np.float32),
        "ln_post.weight": state["ln_post.weight"].detach().cpu().numpy().astype(np.float32),
        "ln_post.bias": state["ln_post.bias"].detach().cpu().numpy().astype(np.float32),
        "proj": state["proj"].detach().cpu().numpy().astype(np.float32),
    }
    for i in range(n_layers):
        prefix = f"transformer.resblocks.{i}"
        for name in (
            "ln_1.weight",
            "ln_1.bias",
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "ln_2.weight",
            "ln_2.bias",
            "mlp.c_fc.weight",
            "mlp.c_fc.bias",
            "mlp.c_proj.weight",
            "mlp.c_proj.bias",
        ):
            key = f"{prefix}.{name}"
            tensors[key] = state[key].detach().cpu().numpy().astype(np.float32)
    save_tensor_family(bundle_dir, family="clip_vision", metadata=metadata, tensors=tensors)



def export_text300m_runtime_family(*, bundle_dir: Path, cache_dir: Path) -> None:
    pt_model = _load_model_with_retry("text300M", cache_dir)
    mlx_tree = map_text300m_weights(pt_model.state_dict())
    tensors: dict[str, np.ndarray] = {}

    def _walk(node: object, prefix: list[object]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, prefix + [key])
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                _walk(value, prefix + [idx])
            return
        key = ".".join(str(part) for part in prefix)
        tensors[key] = np.asarray(node, dtype=np.float32)

    _walk(mlx_tree, [])
    save_tensor_family(
        bundle_dir,
        family="text300m",
        metadata={
            "n_ctx": int(getattr(pt_model, "n_ctx", 1024)),
            "d_latent": int(getattr(pt_model, "d_latent", 1024)),
        },
        tensors=tensors,
    )


def export_image300m_runtime_family(*, bundle_dir: Path, cache_dir: Path) -> None:
    pt_model = _load_model_with_retry("image300M", cache_dir)
    mlx_tree = map_text300m_weights(pt_model.state_dict())
    tensors: dict[str, np.ndarray] = {}

    def _walk(node: object, prefix: list[object]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, prefix + [key])
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                _walk(value, prefix + [idx])
            return
        key = ".".join(str(part) for part in prefix)
        tensors[key] = np.asarray(node, dtype=np.float32)

    _walk(mlx_tree, [])
    save_tensor_family(
        bundle_dir,
        family="image300m",
        metadata={
            "n_ctx": int(getattr(pt_model, "n_ctx", 1024)),
            "d_latent": int(getattr(pt_model, "d_latent", 1024)),
        },
        tensors=tensors,
    )



def export_diffusion_runtime_family(*, bundle_dir: Path, cache_dir: Path) -> None:
    diffusion = diffusion_from_config(load_config("diffusion", cache_dir=str(cache_dir)))
    config = RuntimeDiffusionConfig.from_gaussian_diffusion(diffusion)
    tensors = {
        "alphas_cumprod": config.alphas_cumprod,
        "betas": config.betas,
        "posterior_log_variance_clipped": config.posterior_log_variance_clipped,
        "sqrt_recip_alphas_cumprod": config.sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": config.sqrt_recipm1_alphas_cumprod,
        "posterior_mean_coef1": config.posterior_mean_coef1,
        "posterior_mean_coef2": config.posterior_mean_coef2,
    }
    if config.channel_biases is not None:
        tensors["channel_biases"] = config.channel_biases
    if config.channel_scales is not None:
        tensors["channel_scales"] = config.channel_scales
    save_tensor_family(
        bundle_dir,
        family="diffusion",
        metadata={
            "num_timesteps": config.num_timesteps,
            "model_mean_type": config.model_mean_type,
        },
        tensors=tensors,
    )


def export_decoder_runtime_families(*, bundle_dir: Path, cache_dir: Path) -> None:
    decoder = _load_model_with_retry("decoder", cache_dir)
    decoder_params = MLXChannelsDecoderParams.from_torch_decoder(decoder)
    renderer = MLXSTFMeshRenderer.from_torch_decoder(decoder)

    decoder_tensors: dict[str, np.ndarray] = {}
    for name, projection in decoder_params.projections.items():
        safe = name.replace(".", "__")
        decoder_tensors[f"proj_{safe}_weight_vcd"] = projection.weight_vcd.astype(np.float32)
        decoder_tensors[f"proj_{safe}_bias_vc"] = projection.bias_vc.astype(np.float32)
        decoder_tensors[f"proj_{safe}_ln_weight_c"] = projection.ln_weight_c.astype(np.float32)
        decoder_tensors[f"proj_{safe}_ln_bias_c"] = projection.ln_bias_c.astype(np.float32)
    save_tensor_family(
        bundle_dir,
        family="decoder_params",
        metadata={
            "param_shapes": [(name, list(shape)) for name, shape in decoder_params.param_shapes.items()],
            "latent_ctx": decoder_params.latent_ctx,
            "latent_warp_name": decoder_params.latent_warp_name,
            "latent_warp_coeff1": decoder_params.latent_warp_coeff1,
            "ln_eps": decoder_params.ln_eps,
        },
        tensors=decoder_tensors,
    )

    renderer_tensors = {
        "bbox_min": renderer.bbox_min.astype(np.float32),
        "bbox_max": renderer.bbox_max.astype(np.float32),
    }
    for layer_idx in range(7):
        renderer_tensors[f"mlp_{layer_idx}_bias"] = renderer.static_biases[layer_idx].astype(np.float32)
    for layer_idx in (4, 5, 6):
        renderer_tensors[f"mlp_{layer_idx}_weight"] = renderer.static_weights[layer_idx].astype(np.float32)
    save_tensor_family(
        bundle_dir,
        family="renderer",
        metadata={
            "grid_size": renderer.grid_size,
            "direction_dependent_shape": renderer.direction_dependent_shape,
            "separate_nerf_channels": renderer.separate_nerf_channels,
            "separate_coarse_channels": renderer.separate_coarse_channels,
            "channel_activation": renderer.channel_activation,
            "sdf_activation": renderer.sdf_activation,
            "texture_channels": list(renderer.texture_channels),
        },
        tensors=renderer_tensors,
    )


def prepare_runtime_artifacts(*, bundle_dir: Path, cache_dir: Path, clip_name: str = "ViT-L/14") -> Path:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    export_clip_text_runtime_family(bundle_dir=bundle_dir, cache_dir=cache_dir, clip_name=clip_name)
    export_clip_vision_runtime_family(bundle_dir=bundle_dir, cache_dir=cache_dir, clip_name=clip_name)
    export_text300m_runtime_family(bundle_dir=bundle_dir, cache_dir=cache_dir)
    export_image300m_runtime_family(bundle_dir=bundle_dir, cache_dir=cache_dir)
    export_diffusion_runtime_family(bundle_dir=bundle_dir, cache_dir=cache_dir)
    export_decoder_runtime_families(bundle_dir=bundle_dir, cache_dir=cache_dir)
    return bundle_dir


def main() -> None:
    import argparse
    repo_root = Path(__file__).resolve().parents[4]
    default_cache_dir = repo_root / "shap_e_model_cache"
    default_bundle_dir = repo_root / "runtime_bundle"
    parser = argparse.ArgumentParser(description="Populate the runtime bundle with all required artifact families.")
    parser.add_argument("--bundle-dir", default=default_bundle_dir, help="Directory to output the runtime bundle.")
    parser.add_argument("--cache-dir", default=default_cache_dir, help="Path to shap_e cache directory.")
    args = parser.parse_args()


    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()

    print(f"Preparing all runtime artifacts into: {bundle_dir}")
    prepare_runtime_artifacts(bundle_dir=bundle_dir, cache_dir=cache_dir)
    print("Done!")


if __name__ == "__main__":
    main()
