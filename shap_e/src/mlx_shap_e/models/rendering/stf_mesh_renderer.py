from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import zipfile

import mlx.core as mx
import numpy as np
from shap_e.rendering.mesh import TriMesh

from mlx_shap_e.models.decoder_params import MLXChannelsDecoderParams
from mlx_shap_e.models.rendering.marching_cubes import marching_cubes as marching_cubes_mlx


@dataclass
class MLXMeshResult:
    verts: mx.array
    faces: mx.array
    vertex_channels: Optional[Dict[str, mx.array]] = None

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
class MLXMeshBatchResult:
    meshes: tuple[MLXMeshResult, ...]

    def __len__(self) -> int:
        return len(self.meshes)

    def __iter__(self):
        return iter(self.meshes)

    def tri_meshes(self) -> tuple[TriMesh, ...]:
        return tuple(mesh.tri_mesh() for mesh in self.meshes)


class MLXSTFMeshRenderer:
    """
    Minimal MLX-native STF mesh extraction path for the decoder NeRSTF model.

    This implementation mirrors the mesh-only branch used by Shap-E's STF renderer:
    - evaluates signed distance on a dense bbox grid,
    - pads a negative border,
    - extracts a mesh at isosurface level 0.

    Current scope includes mesh geometry with optional per-vertex color extraction.
    """

    def __init__(
        self,
        *,
        grid_size: int,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        static_weights: Dict[int, np.ndarray],
        static_biases: Dict[int, np.ndarray],
        direction_dependent_shape: bool = True,
        separate_nerf_channels: bool = True,
        separate_coarse_channels: bool = True,
        channel_activation: str = "sigmoid",
        sdf_activation: str = "tanh",
        texture_channels: Tuple[str, ...] = ("R", "G", "B"),
    ):
        self.grid_size = int(grid_size)
        self.bbox_min = bbox_min.astype(np.float32)
        self.bbox_max = bbox_max.astype(np.float32)
        self.static_weights = {k: v.astype(np.float32) for k, v in static_weights.items()}
        self.static_biases = {k: v.astype(np.float32) for k, v in static_biases.items()}
        self.direction_dependent_shape = bool(direction_dependent_shape)
        self.separate_nerf_channels = bool(separate_nerf_channels)
        self.separate_coarse_channels = bool(separate_coarse_channels)
        self.channel_activation = str(channel_activation)
        self.sdf_activation = str(sdf_activation)
        self.texture_channels = tuple(texture_channels)
        self._volume_points_cache: Dict[int, np.ndarray] = {}
        self._volume_points_cache_mx: Dict[int, mx.array] = {}
        self._dir_enc_dim = 3 + 2 * 3 * 8
        self._bbox_min_mx: mx.array | None = None
        self._bbox_max_mx: mx.array | None = None
        self._static_weights_mx: Dict[int, mx.array] = {}
        self._static_biases_mx: Dict[int, mx.array] = {}

    @classmethod
    def from_torch_decoder(cls, decoder) -> "MLXSTFMeshRenderer":
        state = decoder.state_dict()
        static_weights: Dict[int, np.ndarray] = {}
        static_biases: Dict[int, np.ndarray] = {}

        for layer_idx in (4, 5, 6):
            static_weights[layer_idx] = (
                state[f"renderer.nerstf.mlp.{layer_idx}.weight"].detach().cpu().numpy().astype(np.float32)
            )
            static_biases[layer_idx] = (
                state[f"renderer.nerstf.mlp.{layer_idx}.bias"].detach().cpu().numpy().astype(np.float32)
            )

        # Layers 0..3 use latent-projected dynamic weights plus static biases.
        for layer_idx in (0, 1, 2, 3):
            static_biases[layer_idx] = (
                state[f"renderer.nerstf.mlp.{layer_idx}.bias"].detach().cpu().numpy().astype(np.float32)
            )

        bbox_min = decoder.renderer.volume.bbox_min.detach().cpu().numpy().astype(np.float32)
        bbox_max = decoder.renderer.volume.bbox_max.detach().cpu().numpy().astype(np.float32)

        nerstf = getattr(decoder.renderer, "nerstf", None)
        direction_dependent_shape = True
        separate_nerf_channels = True
        separate_coarse_channels = True
        channel_activation = "sigmoid"
        sdf_activation = "tanh"
        texture_channels = ("R", "G", "B")
        if nerstf is not None:
            direction_dependent_shape = getattr(nerstf, "direction_dependent_shape", True)
            separate_nerf_channels = getattr(nerstf, "separate_nerf_channels", True)
            separate_coarse_channels = getattr(nerstf, "separate_coarse_channels", True)
            if hasattr(nerstf, "channel_activation"):
                channel_activation = getattr(getattr(nerstf, "channel_activation"), "__name__", "sigmoid")
            if hasattr(nerstf, "sdf_activation"):
                sdf_activation = getattr(getattr(nerstf, "sdf_activation"), "__name__", "tanh")

        return cls(
            grid_size=int(decoder.renderer.grid_size),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            static_weights=static_weights,
            static_biases=static_biases,
            direction_dependent_shape=direction_dependent_shape,
            separate_nerf_channels=separate_nerf_channels,
            separate_coarse_channels=separate_coarse_channels,
            channel_activation=channel_activation,
            sdf_activation=sdf_activation,
            texture_channels=texture_channels,
        )

    def to_payload(self, *, prefix: str = "renderer_") -> Dict[str, np.ndarray]:
        payload: Dict[str, np.ndarray] = {
            f"{prefix}grid_size": np.array([self.grid_size], dtype=np.int32),
            f"{prefix}bbox_min": self.bbox_min.astype(np.float32),
            f"{prefix}bbox_max": self.bbox_max.astype(np.float32),
            f"{prefix}direction_dependent_shape": np.array([int(self.direction_dependent_shape)], dtype=np.int32),
            f"{prefix}separate_nerf_channels": np.array([int(self.separate_nerf_channels)], dtype=np.int32),
            f"{prefix}separate_coarse_channels": np.array([int(self.separate_coarse_channels)], dtype=np.int32),
            f"{prefix}channel_activation": np.array([self.channel_activation]),
            f"{prefix}sdf_activation": np.array([self.sdf_activation]),
            f"{prefix}texture_channels": np.array(list(self.texture_channels)),
        }
        for layer_idx in (0, 1, 2, 3, 4, 5, 6):
            payload[f"{prefix}mlp_{layer_idx}_bias"] = self.static_biases[layer_idx].astype(np.float32)
        for layer_idx in (4, 5, 6):
            payload[f"{prefix}mlp_{layer_idx}_weight"] = self.static_weights[layer_idx].astype(np.float32)
        return payload

    @classmethod
    def from_payload(cls, payload, *, prefix: str = "renderer_") -> "MLXSTFMeshRenderer":
        static_biases: Dict[int, np.ndarray] = {}
        static_weights: Dict[int, np.ndarray] = {}
        for layer_idx in (0, 1, 2, 3, 4, 5, 6):
            static_biases[layer_idx] = np.array(payload[f"{prefix}mlp_{layer_idx}_bias"], dtype=np.float32)
        for layer_idx in (4, 5, 6):
            static_weights[layer_idx] = np.array(payload[f"{prefix}mlp_{layer_idx}_weight"], dtype=np.float32)
        def _get(key: str, default: np.ndarray) -> np.ndarray:
            return payload[key] if key in payload else default

        direction_dependent_shape = bool(int(_get(f"{prefix}direction_dependent_shape", np.array([1]))[0]))
        separate_nerf_channels = bool(int(_get(f"{prefix}separate_nerf_channels", np.array([1]))[0]))
        separate_coarse_channels = bool(int(_get(f"{prefix}separate_coarse_channels", np.array([1]))[0]))
        channel_activation = str(_get(f"{prefix}channel_activation", np.array(["sigmoid"]))[0])
        sdf_activation = str(_get(f"{prefix}sdf_activation", np.array(["tanh"]))[0])
        texture_channels = tuple(str(x) for x in _get(f"{prefix}texture_channels", np.array(["R", "G", "B"])))
        return cls(
            grid_size=int(payload[f"{prefix}grid_size"][0]),
            bbox_min=np.array(payload[f"{prefix}bbox_min"], dtype=np.float32),
            bbox_max=np.array(payload[f"{prefix}bbox_max"], dtype=np.float32),
            static_weights=static_weights,
            static_biases=static_biases,
            direction_dependent_shape=direction_dependent_shape,
            separate_nerf_channels=separate_nerf_channels,
            separate_coarse_channels=separate_coarse_channels,
            channel_activation=channel_activation,
            sdf_activation=sdf_activation,
            texture_channels=texture_channels,
        )

    def _posenc_nerf(self, x: np.ndarray, *, min_deg: int, max_deg: int) -> np.ndarray:
        if min_deg == max_deg:
            return x
        scales = np.power(2.0, np.arange(min_deg, max_deg, dtype=x.dtype))
        xb = (x[:, None, :] * scales[None, :, None]).reshape(x.shape[0], -1)
        emb = np.sin(np.concatenate([xb, xb + (np.pi / 2.0)], axis=-1))
        return np.concatenate([x, emb], axis=-1).astype(np.float32)

    def _posenc_nerf_mx(self, x: mx.array, *, min_deg: int, max_deg: int) -> mx.array:
        if min_deg == max_deg:
            return x.astype(mx.float32)
        scales = mx.power(
            mx.array(2.0, dtype=mx.float32),
            mx.arange(min_deg, max_deg, dtype=mx.float32),
        )
        leading_shape = tuple(int(v) for v in x.shape[:-1])
        xb = (mx.expand_dims(x, axis=-2) * scales.reshape((1,) * len(leading_shape) + (-1, 1))).reshape(
            leading_shape + (-1,)
        )
        emb = mx.sin(mx.concatenate([xb, xb + (np.pi / 2.0)], axis=-1))
        return mx.concatenate([x, emb], axis=-1).astype(mx.float32)

    def _swish(self, x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -60.0, 60.0)
        return x * (1.0 / (1.0 + np.exp(-x_clip)))

    def _swish_mx(self, x: mx.array) -> mx.array:
        x_clip = mx.clip(x, -60.0, 60.0)
        return x * (1.0 / (1.0 + mx.exp(-x_clip)))

    def _meta_linear(self, h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        # h: [N, in], w: [out, in], b: [out]
        return (h @ w.T) + b

    def _meta_linear_mx(self, h: mx.array, w: mx.array, b: mx.array) -> mx.array:
        if h.ndim == 2 and w.ndim == 2:
            return mx.matmul(h, mx.transpose(w, (1, 0))) + b
        if h.ndim == 3 and w.ndim == 3:
            return mx.matmul(h, mx.transpose(w, (0, 2, 1))) + b
        raise ValueError(f"Unsupported meta-linear shapes: h={h.shape}, w={w.shape}, b={b.shape}")

    def signed_distance(
        self,
        positions: np.ndarray,
        dynamic_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        h = self._nerstf_forward(positions, dynamic_params)
        return self._apply_activation(h[:, :1], self.sdf_activation).astype(np.float32)

    def texture_channels_from_positions(
        self,
        positions: np.ndarray,
        dynamic_params: Dict[str, np.ndarray],
    ) -> np.ndarray:
        h = self._nerstf_forward(positions, dynamic_params)
        start, end = self._stf_channel_slice()
        channels = h[:, start:end]
        channels = self._apply_activation(channels, self.channel_activation)
        return np.clip(channels.astype(np.float32), 0.0, 1.0)

    def signed_distance_mx(
        self,
        positions: mx.array,
        dynamic_params: Dict[str, np.ndarray | mx.array],
    ) -> mx.array:
        h = self._nerstf_forward_mx(positions, dynamic_params)
        return self._apply_activation_mx(h[..., :1], self.sdf_activation).astype(mx.float32)

    def texture_channels_from_positions_mx(
        self,
        positions: mx.array,
        dynamic_params: Dict[str, np.ndarray | mx.array],
    ) -> mx.array:
        h = self._nerstf_forward_mx(positions, dynamic_params)
        start, end = self._stf_channel_slice()
        channels = h[..., start:end]
        channels = self._apply_activation_mx(channels, self.channel_activation)
        return mx.clip(channels.astype(mx.float32), 0.0, 1.0)

    def _volume_query_points(self, grid_size: int) -> np.ndarray:
        cached = self._volume_points_cache.get(grid_size)
        if cached is not None:
            return cached
        indices = np.arange(grid_size**3, dtype=np.int64)
        zs = indices % grid_size
        ys = (indices // grid_size) % grid_size
        xs = (indices // (grid_size**2)) % grid_size
        combined = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        points = (combined / float(grid_size - 1)) * (self.bbox_max - self.bbox_min) + self.bbox_min
        self._volume_points_cache[grid_size] = points.astype(np.float32)
        return self._volume_points_cache[grid_size]

    def _volume_query_points_mx(self, grid_size: int) -> mx.array:
        cached = self._volume_points_cache_mx.get(grid_size)
        if cached is not None:
            return cached
        indices = mx.arange(grid_size**3, dtype=mx.int32)
        zs = indices % grid_size
        ys = (indices // grid_size) % grid_size
        xs = (indices // (grid_size**2)) % grid_size
        combined = mx.stack([xs, ys, zs], axis=1).astype(mx.float32)
        points = (
            combined / float(grid_size - 1)
        ) * (self._bbox_max_array() - self._bbox_min_array()) + self._bbox_min_array()
        self._volume_points_cache_mx[grid_size] = points.astype(mx.float32)
        return self._volume_points_cache_mx[grid_size]

    def extract_sdf_fields(
        self,
        dynamic_params: Dict[str, np.ndarray | mx.array],
        *,
        query_batch_size: int = 4096,
        grid_size: Optional[int] = None,
    ) -> mx.array:
        """
        Evaluate the unpadded SDF volume for each batch item.

        Returns:
            float32 field with shape [B, G, G, G] (before the -1 border pad used by MC).
        """
        active_grid_size = int(grid_size) if grid_size is not None else self.grid_size
        if active_grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {active_grid_size}")

        dynamic_params_mx = {name: self._dynamic_weight_array(value) for name, value in dynamic_params.items()}
        batch_size = self._dynamic_batch_size(dynamic_params_mx)
        points = self._volume_query_points_mx(active_grid_size)
        sdf_batches = []
        for i in range(0, int(points.shape[0]), int(query_batch_size)):
            j = min(i + int(query_batch_size), int(points.shape[0]))
            sdf_chunk = self.signed_distance_mx(points[i:j], dynamic_params_mx)
            if sdf_chunk.ndim == 2:
                sdf_chunk = mx.expand_dims(sdf_chunk, axis=0)
            sdf_batches.append(sdf_chunk)
        sdf_values = (
            mx.concatenate(sdf_batches, axis=1)
            if sdf_batches
            else mx.zeros((batch_size, 0, 1), dtype=mx.float32)
        )
        return sdf_values.reshape((batch_size, active_grid_size, active_grid_size, active_grid_size)).astype(mx.float32)

    def extract_mesh(
        self,
        dynamic_params: Dict[str, np.ndarray | mx.array],
        *,
        query_batch_size: int = 4096,
        include_vertex_colors: bool = False,
        grid_size: Optional[int] = None,
    ) -> MLXMeshResult:
        meshes = self.extract_meshes(
            dynamic_params,
            query_batch_size=query_batch_size,
            include_vertex_colors=include_vertex_colors,
            grid_size=grid_size,
        )
        if len(meshes) != 1:
            raise ValueError(
                f"extract_mesh expected a single latent worth of params, got batch size {len(meshes)}. "
                "Use extract_meshes(...) for batched decode."
            )
        return meshes.meshes[0]

    def extract_meshes(
        self,
        dynamic_params: Dict[str, np.ndarray | mx.array],
        *,
        query_batch_size: int = 4096,
        include_vertex_colors: bool = False,
        grid_size: Optional[int] = None,
    ) -> MLXMeshBatchResult:
        fields = self.extract_sdf_fields(
            dynamic_params,
            query_batch_size=query_batch_size,
            grid_size=grid_size,
        )
        batch_size = int(fields.shape[0])
        dynamic_params_mx = {name: self._dynamic_weight_array(value) for name, value in dynamic_params.items()}

        meshes: list[MLXMeshResult] = []
        bbox_min = self._bbox_min_array()
        bbox_size = self._bbox_max_array() - bbox_min
        for batch_idx in range(batch_size):
            # Match Shap-E behavior by forcing a negative border around the SDF.
            full = mx.pad(fields[batch_idx], [(1, 1), (1, 1), (1, 1)], mode="constant", constant_values=-1.0).astype(
                mx.float32
            )
            vmin = float(mx.min(full))
            vmax = float(mx.max(full))
            if not (vmin <= 0.0 <= vmax):
                meshes.append(self._empty_mesh_result())
                continue

            verts_world, faces = marching_cubes_mlx(full, bbox_min, bbox_size)
            if int(faces.shape[0]) == 0:
                meshes.append(self._empty_mesh_result())
                continue

            vertex_channels = None
            if include_vertex_colors and int(verts_world.shape[0]) > 0:
                params_i = self._dynamic_params_batch_item(dynamic_params_mx, batch_idx)
                channels = self.texture_channels_from_positions_mx(verts_world, params_i)
                vertex_channels = {
                    name: channels[:, i].astype(mx.float32) for i, name in enumerate(self.texture_channels)
                }

            meshes.append(
                MLXMeshResult(
                    verts=verts_world.astype(mx.float32),
                    faces=faces.astype(mx.int32),
                    vertex_channels=vertex_channels,
                )
            )

        return MLXMeshBatchResult(meshes=tuple(meshes))

    def _nerstf_forward(self, positions: np.ndarray, dynamic_params: Dict[str, np.ndarray]) -> np.ndarray:
        pos = positions.astype(np.float32)
        h = self._posenc_nerf(pos, min_deg=0, max_deg=15)

        # Dynamic meta layers 0..3.
        for layer_idx in (0, 1, 2, 3):
            w = dynamic_params[f"nerstf.mlp.{layer_idx}.weight"][0].astype(np.float32)
            b = self.static_biases[layer_idx]
            h = self._meta_linear(h, w, b)
            h = self._swish(h)

        # Direction insertion at layer 4 (direction is None in mesh path -> zeros).
        dir_enc = np.zeros((pos.shape[0], self._dir_enc_dim), dtype=np.float32)
        h = np.concatenate([h, dir_enc], axis=-1).astype(np.float32)

        # Static layers 4..6.
        for layer_idx in (4, 5, 6):
            w = self.static_weights[layer_idx]
            b = self.static_biases[layer_idx]
            h = (h @ w.T) + b
            if layer_idx < 6:
                h = self._swish(h)
        return h.astype(np.float32)

    def _stf_channel_slice(self) -> Tuple[int, int]:
        if self.direction_dependent_shape:
            if self.separate_coarse_channels:
                return (3, 6)
            return (2, 5)
        return (0, 3)

    def _apply_activation(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        if name == "tanh":
            return np.tanh(x)
        if name == "identity":
            return x
        raise ValueError(f"Unsupported activation: {name}")

    def _apply_activation_mx(self, x: mx.array, name: str) -> mx.array:
        if name == "sigmoid":
            return 1.0 / (1.0 + mx.exp(-x))
        if name == "tanh":
            return mx.tanh(x)
        if name == "identity":
            return x
        raise ValueError(f"Unsupported activation: {name}")

    def _bbox_min_array(self) -> mx.array:
        if self._bbox_min_mx is None:
            self._bbox_min_mx = mx.array(self.bbox_min.astype(np.float32))
        return self._bbox_min_mx

    def _bbox_max_array(self) -> mx.array:
        if self._bbox_max_mx is None:
            self._bbox_max_mx = mx.array(self.bbox_max.astype(np.float32))
        return self._bbox_max_mx

    def _static_weight_array(self, layer_idx: int) -> mx.array:
        cached = self._static_weights_mx.get(layer_idx)
        if cached is None:
            cached = mx.array(self.static_weights[layer_idx].astype(np.float32))
            self._static_weights_mx[layer_idx] = cached
        return cached

    def _static_bias_array(self, layer_idx: int) -> mx.array:
        cached = self._static_biases_mx.get(layer_idx)
        if cached is None:
            cached = mx.array(self.static_biases[layer_idx].astype(np.float32))
            self._static_biases_mx[layer_idx] = cached
        return cached

    def _dynamic_weight_array(self, value: np.ndarray | mx.array) -> mx.array:
        if value.__class__.__module__.startswith("mlx.core"):
            return value.astype(mx.float32)
        return mx.array(np.asarray(value, dtype=np.float32))

    def _dynamic_batch_size(self, dynamic_params: Dict[str, mx.array]) -> int:
        for value in dynamic_params.values():
            if value.ndim == 0:
                continue
            return int(value.shape[0])
        raise ValueError("dynamic_params must contain at least one batched tensor.")

    def _dynamic_params_batch_item(
        self,
        dynamic_params: Dict[str, mx.array],
        batch_idx: int,
    ) -> Dict[str, mx.array]:
        return {name: value[batch_idx : batch_idx + 1] for name, value in dynamic_params.items()}

    def _empty_mesh_result(self) -> MLXMeshResult:
        return MLXMeshResult(
            verts=mx.zeros((0, 3), dtype=mx.float32),
            faces=mx.zeros((0, 3), dtype=mx.int32),
        )

    def _nerstf_forward_mx(
        self,
        positions: mx.array,
        dynamic_params: Dict[str, np.ndarray | mx.array],
    ) -> mx.array:
        dynamic_params_mx = {name: self._dynamic_weight_array(value) for name, value in dynamic_params.items()}
        batch_size = self._dynamic_batch_size(dynamic_params_mx)

        pos = positions.astype(mx.float32)
        squeeze_output = pos.ndim == 2 and batch_size == 1
        if pos.ndim == 2:
            pos = mx.expand_dims(pos, axis=0)
            if batch_size != 1:
                pos = mx.broadcast_to(pos, (batch_size, int(pos.shape[1]), int(pos.shape[2])))
        elif pos.ndim == 3:
            if int(pos.shape[0]) == 1 and batch_size != 1:
                pos = mx.broadcast_to(pos, (batch_size, int(pos.shape[1]), int(pos.shape[2])))
            elif int(pos.shape[0]) != batch_size:
                raise ValueError(
                    f"positions batch {int(pos.shape[0])} does not match dynamic_params batch {batch_size}."
                )
        else:
            raise ValueError(f"positions must be [N, 3] or [B, N, 3], got {pos.shape}")

        h = self._posenc_nerf_mx(pos, min_deg=0, max_deg=15)

        for layer_idx in (0, 1, 2, 3):
            w = dynamic_params_mx[f"nerstf.mlp.{layer_idx}.weight"]
            b = self._static_bias_array(layer_idx).reshape((1, 1, -1))
            h = self._meta_linear_mx(h, w, b)
            h = self._swish_mx(h)

        dir_enc = mx.zeros((int(pos.shape[0]), int(pos.shape[1]), self._dir_enc_dim), dtype=mx.float32)
        h = mx.concatenate([h, dir_enc], axis=-1).astype(mx.float32)

        for layer_idx in (4, 5, 6):
            w = self._static_weight_array(layer_idx)
            b = self._static_bias_array(layer_idx).reshape((1, 1, -1))
            h = mx.matmul(h, mx.transpose(w, (1, 0))) + b
            if layer_idx < 6:
                h = self._swish_mx(h)
        h = h.astype(mx.float32)
        if squeeze_output:
            return h[0]
        return h


def save_mlx_decoder_artifact(artifact_path: Path, decoder) -> None:
    print(f"Exporting MLX decoder artifact to {artifact_path}...")
    artifact_path = Path(artifact_path).expanduser().resolve()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_decoder = MLXChannelsDecoderParams.from_torch_decoder(decoder)
    renderer = MLXSTFMeshRenderer.from_torch_decoder(decoder)
    payload: Dict[str, np.ndarray] = {
        "artifact_version": np.array([1], dtype=np.int32),
        "artifact_type": np.array(["mlx_stf_decoder_mesh_v1"]),
    }
    payload.update(mlx_decoder.to_payload(prefix="decoder_"))
    payload.update(renderer.to_payload(prefix="renderer_"))
    np.savez_compressed(artifact_path, **payload)


def load_mlx_decoder_artifact(artifact_path: Path) -> Tuple[MLXChannelsDecoderParams, MLXSTFMeshRenderer]:
    artifact_path = Path(artifact_path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"MLX decoder artifact not found: {artifact_path}. "
            "Generate it with: python shap_e/src/export_mlx_decoder_artifact.py"
        )
    data = np.load(artifact_path)
    required_keys = (
        "decoder_param_shapes_json",
        "decoder_latent_ctx",
        "renderer_grid_size",
        "renderer_bbox_min",
        "renderer_bbox_max",
    )
    missing = [k for k in required_keys if k not in data]
    if missing:
        hint = ""
        if "final_latent_flat" in data or "latent_flat" in data:
            hint = " This looks like a latent run artifact, not a decoder mesh artifact."
        raise ValueError(
            f"Invalid MLX decoder artifact at {artifact_path}. Missing keys: {missing}.{hint} "
            "Re-export with: python shap_e/src/export_mlx_decoder_artifact.py "
            "--output shap_e/outputs/artifacts/mlx_decoder_mesh_v1.npz"
        )
    if "artifact_type" in data:
        artifact_type = str(data["artifact_type"][0])
        if artifact_type != "mlx_stf_decoder_mesh_v1":
            raise ValueError(f"Unsupported artifact type: {artifact_type}")
    mlx_decoder = MLXChannelsDecoderParams.from_payload(data, prefix="decoder_")
    renderer = MLXSTFMeshRenderer.from_payload(data, prefix="renderer_")
    return mlx_decoder, renderer


def decode_latent_mesh_mlx_renderer(
    decoder,
    latent: np.ndarray | mx.array,
    mlx_decoder: MLXChannelsDecoderParams,
    *,
    query_batch_size: int = 4096,
    include_vertex_colors: bool = False,
    grid_size_override: Optional[int] = None,
) -> MLXMeshResult:
    renderer = MLXSTFMeshRenderer.from_torch_decoder(decoder)
    latent_mx = latent.astype(mx.float32) if latent.__class__.__module__.startswith("mlx.core") else mx.array(
        np.asarray(latent, dtype=np.float32)
    )
    if latent_mx.ndim == 1:
        latent_mx = mx.expand_dims(latent_mx, axis=0)
    elif latent_mx.ndim != 2:
        raise ValueError(f"latent must be [D] or [B, D], got {latent_mx.shape}")
    params_mx = mlx_decoder.bottleneck_to_params_mx(latent_mx)
    return renderer.extract_mesh(
        params_mx,
        query_batch_size=query_batch_size,
        include_vertex_colors=include_vertex_colors,
        grid_size=grid_size_override,
    )


def decode_latent_mesh_mlx_renderer_artifact(
    artifact_path: Path,
    latent: np.ndarray | mx.array,
    *,
    query_batch_size: int = 4096,
    regenerate_fn: Callable[[], None] | None = None,
    include_vertex_colors: bool = False,
    grid_size_override: Optional[int] = None,
) -> MLXMeshResult:
    try:
        mlx_decoder, renderer = load_mlx_decoder_artifact(artifact_path)
    except zipfile.BadZipFile:
        if regenerate_fn is None:
            raise
        if artifact_path.exists():
            artifact_path.unlink()
        regenerate_fn()
        try:
            mlx_decoder, renderer = load_mlx_decoder_artifact(artifact_path)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"Decoder artifact is still corrupted after regeneration: {artifact_path}. "
                "Delete it and rerun the pipeline."
            ) from exc
    latent_mx = latent.astype(mx.float32) if latent.__class__.__module__.startswith("mlx.core") else mx.array(
        np.asarray(latent, dtype=np.float32)
    )
    if latent_mx.ndim == 1:
        latent_mx = mx.expand_dims(latent_mx, axis=0)
    elif latent_mx.ndim != 2:
        raise ValueError(f"latent must be [D] or [B, D], got {latent_mx.shape}")
    params_mx = mlx_decoder.bottleneck_to_params_mx(latent_mx)
    return renderer.extract_mesh(
        params_mx,
        query_batch_size=query_batch_size,
        include_vertex_colors=include_vertex_colors,
        grid_size=grid_size_override,
    )
