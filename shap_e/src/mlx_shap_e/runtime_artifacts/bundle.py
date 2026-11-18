from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np

from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams, _ProjWeights
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextArtifact
from mlx_shap_e.models.rendering.stf_mesh_renderer import MLXSTFMeshRenderer
from mlx_shap_e.models.artifacts.text300m_artifact import Text300MArtifact
from mlx_shap_e.runtime_artifacts.manifest import load_manifest, manifest_path
from mlx_shap_e.runtime_core.diffusion import RuntimeDiffusionConfig


class RuntimeArtifactBundle:
    def __init__(self, bundle_dir: str | Path):
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.manifest = load_manifest(self.bundle_dir)

    @property
    def path(self) -> Path:
        return self.bundle_dir

    @property
    def manifest_file(self) -> Path:
        return manifest_path(self.bundle_dir)

    def family_version(self, family: str) -> int:
        return int(self.require_family(family).version)

    def require_family(self, family: str):
        if family not in self.manifest.families:
            available = ", ".join(sorted(self.manifest.families.keys()))
            raise FileNotFoundError(
                f"Runtime artifact family '{family}' not found in {self.manifest_file}. "
                f"Available families: {available or '<none>'}."
            )
        return self.manifest.families[family]

    def load_family_tensors(self, family: str) -> Dict[str, np.ndarray]:
        family_manifest = self.require_family(family)
        tensors: Dict[str, np.ndarray] = {}
        for tensor_name, entry in family_manifest.tensors.items():
            tensor_path = self.bundle_dir / entry.path
            if not tensor_path.exists():
                raise FileNotFoundError(
                    f"Tensor blob missing for family '{family}': {tensor_path}. "
                    "Re-export the runtime artifacts offline."
                )
            array = np.fromfile(tensor_path, dtype=np.dtype(entry.dtype))
            tensors[tensor_name] = array.reshape(entry.shape)
        return tensors

    def load_clip_text_artifact(self) -> MlxClipTextArtifact:
        family = self.require_family("clip_text")
        metadata = family.metadata
        tensors = self.load_family_tensors("clip_text")
        return MlxClipTextArtifact(
            context_length=int(metadata["context_length"]),
            transformer_width=int(metadata["transformer_width"]),
            transformer_heads=int(metadata["transformer_heads"]),
            transformer_layers=int(metadata["transformer_layers"]),
            state=tensors,
        )

    def load_text300m_artifact(self) -> Text300MArtifact:
        family = self.require_family("text300m")
        metadata = family.metadata
        tensors = self.load_family_tensors("text300m")
        return Text300MArtifact(
            n_ctx=int(metadata["n_ctx"]),
            d_latent=int(metadata["d_latent"]),
            state=tensors,
        )

    def load_diffusion_config(self) -> RuntimeDiffusionConfig:
        family = self.require_family("diffusion")
        metadata = family.metadata
        tensors = self.load_family_tensors("diffusion")
        return RuntimeDiffusionConfig(
            alphas_cumprod=tensors["alphas_cumprod"],
            betas=tensors["betas"],
            posterior_log_variance_clipped=tensors["posterior_log_variance_clipped"],
            sqrt_recip_alphas_cumprod=tensors["sqrt_recip_alphas_cumprod"],
            sqrt_recipm1_alphas_cumprod=tensors["sqrt_recipm1_alphas_cumprod"],
            posterior_mean_coef1=tensors["posterior_mean_coef1"],
            posterior_mean_coef2=tensors["posterior_mean_coef2"],
            num_timesteps=int(metadata["num_timesteps"]),
            model_mean_type=str(metadata.get("model_mean_type", "epsilon")),
            channel_biases=tensors.get("channel_biases"),
            channel_scales=tensors.get("channel_scales"),
        )

    def load_decoder_params(self) -> MLXChannelsDecoderParams:
        family = self.require_family("decoder_params")
        metadata = family.metadata
        tensors = self.load_family_tensors("decoder_params")
        param_shapes = OrderedDict(
            (name, tuple(int(v) for v in shape))
            for name, shape in metadata["param_shapes"]
        )
        projections: Dict[str, _ProjWeights] = {}
        for name in param_shapes.keys():
            safe = name.replace(".", "__")
            projections[name] = _ProjWeights(
                weight_vcd=tensors[f"proj_{safe}_weight_vcd"],
                bias_vc=tensors[f"proj_{safe}_bias_vc"],
                ln_weight_c=tensors[f"proj_{safe}_ln_weight_c"],
                ln_bias_c=tensors[f"proj_{safe}_ln_bias_c"],
            )
        return MLXChannelsDecoderParams(
            param_shapes=param_shapes,
            latent_ctx=int(metadata["latent_ctx"]),
            latent_warp_name=str(metadata["latent_warp_name"]),
            latent_warp_coeff1=float(metadata["latent_warp_coeff1"]),
            projections=projections,
            ln_eps=float(metadata.get("ln_eps", 1e-5)),
        )

    def load_renderer(self) -> MLXSTFMeshRenderer:
        family = self.require_family("renderer")
        metadata = family.metadata
        tensors = self.load_family_tensors("renderer")
        static_biases = {layer_idx: tensors[f"mlp_{layer_idx}_bias"] for layer_idx in range(7)}
        static_weights = {layer_idx: tensors[f"mlp_{layer_idx}_weight"] for layer_idx in (4, 5, 6)}
        return MLXSTFMeshRenderer(
            grid_size=int(metadata["grid_size"]),
            bbox_min=tensors["bbox_min"],
            bbox_max=tensors["bbox_max"],
            static_weights=static_weights,
            static_biases=static_biases,
            direction_dependent_shape=bool(metadata.get("direction_dependent_shape", True)),
            separate_nerf_channels=bool(metadata.get("separate_nerf_channels", True)),
            separate_coarse_channels=bool(metadata.get("separate_coarse_channels", True)),
            channel_activation=str(metadata.get("channel_activation", "sigmoid")),
            sdf_activation=str(metadata.get("sdf_activation", "tanh")),
            texture_channels=tuple(metadata.get("texture_channels", ["R", "G", "B"])),
        )
