from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple

import mlx.core as mx
import numpy as np
from shap_e.util.collections import AttrDict


def _flatten_param_shapes(param_shapes: Dict[str, Tuple[int, ...]]) -> "OrderedDict[str, Tuple[int, int]]":
    return OrderedDict(
        (name, (int(np.prod(shape)) // shape[-1], shape[-1]))
        for name, shape in param_shapes.items()
    )


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")


@dataclass
class _ProjWeights:
    weight_vcd: np.ndarray
    bias_vc: np.ndarray
    ln_weight_c: np.ndarray
    ln_bias_c: np.ndarray


class MLXChannelsDecoderParams:
    """Numpy/MLX-style mirror of ChannelsDecoder.bottleneck_to_params for Shap-E decoder."""

    def __init__(
        self,
        *,
        param_shapes: Dict[str, Tuple[int, ...]],
        latent_ctx: int,
        latent_warp_name: str,
        latent_warp_coeff1: float,
        projections: Dict[str, _ProjWeights],
        ln_eps: float = 1e-5,
    ):
        self.param_shapes = OrderedDict(param_shapes)
        self.flat_shapes = _flatten_param_shapes(self.param_shapes)
        self.latent_ctx = latent_ctx
        self.latent_warp_name = latent_warp_name
        self.latent_warp_coeff1 = latent_warp_coeff1
        self.projections = projections
        self.ln_eps = ln_eps

    @classmethod
    def from_torch_decoder(cls, decoder) -> "MLXChannelsDecoderParams":
        latent_warp = decoder.latent_warp
        latent_warp_name = "identity"
        latent_warp_coeff1 = 1.0
        if latent_warp.__class__.__name__ == "Tan2LatentWarp":
            latent_warp_name = "tan2"
            latent_warp_coeff1 = float(latent_warp.coeff1)

        proj_mod = decoder.params_proj.projections
        projections: Dict[str, _ProjWeights] = {}
        for k, (vectors, channels) in _flatten_param_shapes(decoder.param_shapes).items():
            mod = proj_mod[_sanitize_name(k)]
            w = mod.proj.weight.detach().cpu().numpy().astype(np.float32).reshape(vectors, channels, -1)
            b = mod.proj.bias.detach().cpu().numpy().astype(np.float32).reshape(vectors, channels)
            ln_w = mod.norm.weight.detach().cpu().numpy().astype(np.float32)
            ln_b = mod.norm.bias.detach().cpu().numpy().astype(np.float32)
            projections[k] = _ProjWeights(
                weight_vcd=w,
                bias_vc=b,
                ln_weight_c=ln_w,
                ln_bias_c=ln_b,
            )

        return cls(
            param_shapes=decoder.param_shapes,
            latent_ctx=decoder.latent_ctx,
            latent_warp_name=latent_warp_name,
            latent_warp_coeff1=latent_warp_coeff1,
            projections=projections,
        )

    def to_payload(self, *, prefix: str = "decoder_") -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {}
        payload[f"{prefix}param_shapes_json"] = np.array(
            json.dumps([(k, list(v)) for k, v in self.param_shapes.items()])
        )
        payload[f"{prefix}latent_ctx"] = np.array([self.latent_ctx], dtype=np.int32)
        payload[f"{prefix}latent_warp_name"] = np.array([self.latent_warp_name])
        payload[f"{prefix}latent_warp_coeff1"] = np.array([self.latent_warp_coeff1], dtype=np.float32)
        payload[f"{prefix}ln_eps"] = np.array([self.ln_eps], dtype=np.float32)
        for k, proj in self.projections.items():
            safe = _sanitize_name(k)
            payload[f"{prefix}proj_{safe}_weight_vcd"] = proj.weight_vcd.astype(np.float32)
            payload[f"{prefix}proj_{safe}_bias_vc"] = proj.bias_vc.astype(np.float32)
            payload[f"{prefix}proj_{safe}_ln_weight_c"] = proj.ln_weight_c.astype(np.float32)
            payload[f"{prefix}proj_{safe}_ln_bias_c"] = proj.ln_bias_c.astype(np.float32)
        return payload

    @classmethod
    def from_payload(cls, payload, *, prefix: str = "decoder_") -> "MLXChannelsDecoderParams":
        items = json.loads(str(payload[f"{prefix}param_shapes_json"].item()))
        param_shapes = OrderedDict((k, tuple(int(x) for x in shape)) for k, shape in items)
        projections: Dict[str, _ProjWeights] = {}
        for k in param_shapes.keys():
            safe = _sanitize_name(k)
            projections[k] = _ProjWeights(
                weight_vcd=np.array(payload[f"{prefix}proj_{safe}_weight_vcd"], dtype=np.float32),
                bias_vc=np.array(payload[f"{prefix}proj_{safe}_bias_vc"], dtype=np.float32),
                ln_weight_c=np.array(payload[f"{prefix}proj_{safe}_ln_weight_c"], dtype=np.float32),
                ln_bias_c=np.array(payload[f"{prefix}proj_{safe}_ln_bias_c"], dtype=np.float32),
            )
        return cls(
            param_shapes=param_shapes,
            latent_ctx=int(payload[f"{prefix}latent_ctx"][0]),
            latent_warp_name=str(payload[f"{prefix}latent_warp_name"][0]),
            latent_warp_coeff1=float(payload[f"{prefix}latent_warp_coeff1"][0]),
            projections=projections,
            ln_eps=float(payload[f"{prefix}ln_eps"][0]),
        )

    def _unwarp(self, x: np.ndarray) -> np.ndarray:
        if self.latent_warp_name == "identity":
            return x
        if self.latent_warp_name == "tan2":
            scale = np.tan(np.tan(1.0) * self.latent_warp_coeff1)
            return np.arctan(np.arctan(x * scale) / self.latent_warp_coeff1).astype(np.float32)
        raise NotImplementedError(f"Unsupported latent warp: {self.latent_warp_name}")

    def _unwarp_mx(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        if self.latent_warp_name == "identity":
            return x
        if self.latent_warp_name == "tan2":
            scale = float(np.tan(np.tan(1.0) * self.latent_warp_coeff1))
            return mx.arctan(mx.arctan(x * scale) / float(self.latent_warp_coeff1)).astype(mx.float32)
        raise NotImplementedError(f"Unsupported latent warp: {self.latent_warp_name}")

    def bottleneck_to_params_numpy(self, vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Args:
            vector: [B, d_latent] flat latent.
        Returns:
            Dict of parameter tensors matching decoder.param_shapes.
        """
        if vector.ndim != 2:
            raise ValueError(f"Expected latent [B, D], got {vector.shape}")

        bsz = vector.shape[0]
        x = self._unwarp(vector.astype(np.float32)).reshape(bsz, self.latent_ctx, -1)

        out: Dict[str, np.ndarray] = {}
        start = 0
        for k, shape in self.param_shapes.items():
            vectors, channels = self.flat_shapes[k]
            end = start + vectors
            x_bvd = x[:, start:end, :]
            start = end

            proj = self.projections[k]
            h = np.einsum("bvd,vcd->bvc", x_bvd, proj.weight_vcd)

            # LayerNorm over channels dimension (equivalent to PyTorch LayerNorm(channels)).
            mean = np.mean(h, axis=-1, keepdims=True)
            var = np.mean((h - mean) ** 2, axis=-1, keepdims=True)
            h = (h - mean) / np.sqrt(var + self.ln_eps)
            h = h * proj.ln_weight_c.reshape(1, 1, channels) + proj.ln_bias_c.reshape(1, 1, channels)

            h = h + proj.bias_vc.reshape(1, vectors, channels)
            out[k] = h.reshape((bsz, *shape)).astype(np.float32)

        return out

    def bottleneck_to_params_mx(self, vector: mx.array) -> Dict[str, mx.array]:
        """
        Args:
            vector: [B, d_latent] flat latent as MLX tensor.
        Returns:
            Dict of parameter tensors matching decoder.param_shapes.
        """
        if vector.ndim != 2:
            raise ValueError(f"Expected latent [B, D], got {vector.shape}")

        bsz = int(vector.shape[0])
        x = self._unwarp_mx(vector).reshape((bsz, self.latent_ctx, -1))

        out: Dict[str, mx.array] = {}
        start = 0
        for k, shape in self.param_shapes.items():
            vectors, channels = self.flat_shapes[k]
            end = start + vectors
            x_bvd = x[:, start:end, :]
            start = end

            proj = self.projections[k]
            weight_vcd = mx.array(proj.weight_vcd.astype(np.float32))
            bias_vc = mx.array(proj.bias_vc.astype(np.float32))
            ln_weight_c = mx.array(proj.ln_weight_c.astype(np.float32))
            ln_bias_c = mx.array(proj.ln_bias_c.astype(np.float32))

            h = mx.einsum("bvd,vcd->bvc", x_bvd, weight_vcd)
            mean = mx.mean(h, axis=-1, keepdims=True)
            var = mx.mean((h - mean) * (h - mean), axis=-1, keepdims=True)
            h = (h - mean) * mx.rsqrt(var + float(self.ln_eps))
            h = h * ln_weight_c.reshape((1, 1, channels)) + ln_bias_c.reshape((1, 1, channels))
            h = h + bias_vc.reshape((1, vectors, channels))
            out[k] = h.reshape((bsz, *shape)).astype(mx.float32)

        return out

    def bottleneck_to_framework_params(
        self,
        vector: np.ndarray,
        *,
        reference_tensor,
    ) -> AttrDict:
        params_np = self.bottleneck_to_params_numpy(vector)
        params = AttrDict()
        for k, v in params_np.items():
            params[k] = reference_tensor.new_tensor(v).float()
        return params
