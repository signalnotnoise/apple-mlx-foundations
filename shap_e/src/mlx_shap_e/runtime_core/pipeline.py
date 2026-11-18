from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from mlx_shap_e.models.diffusion.karras_sampler import karras_sample_heun_mlx
from mlx_shap_e.models.embeddings.clip_text_embedder import MlxClipTextEmbedder
from mlx_shap_e.models.text300m import Text300MTransformer
from mlx_shap_e.runtime_artifacts.bundle import RuntimeArtifactBundle
from mlx_shap_e.runtime_core.types import ArtifactVersionInfo, BoundingBox, LatentResult, MeshBatchResult, MeshResult


class MLXShapERuntime:
    def __init__(
        self,
        artifact_bundle: str | Path | RuntimeArtifactBundle,
        *,
        tokenizer_bpe_path: str | Path | None = None,
    ):
        bundle = artifact_bundle if isinstance(artifact_bundle, RuntimeArtifactBundle) else RuntimeArtifactBundle(artifact_bundle)
        self.bundle = bundle
        self.artifact_info = ArtifactVersionInfo(
            artifact_version=int(bundle.manifest.artifact_version),
            manifest_path=str(bundle.manifest_file),
            family_versions={name: family.version for name, family in bundle.manifest.families.items()},
        )
        self.clip_artifact = bundle.load_clip_text_artifact()
        self.text300m_artifact = bundle.load_text300m_artifact()
        self.diffusion = bundle.load_diffusion_config()
        self.decoder_params = bundle.load_decoder_params()
        self.renderer = bundle.load_renderer()

        self.embedder = MlxClipTextEmbedder(self.clip_artifact, tokenizer_bpe_path=tokenizer_bpe_path)
        self.model = Text300MTransformer()
        self.model.update(self.text300m_artifact.to_mlx_tree())

    @staticmethod
    def _is_mx_array(value: object) -> bool:
        return value.__class__.__module__.startswith("mlx.core")

    @classmethod
    def _to_mx_f32(cls, value) -> mx.array:
        if cls._is_mx_array(value):
            return value.astype(mx.float32)
        return mx.array(value, dtype=mx.float32)

    def embed_text_mx(self, prompts: Iterable[str]) -> mx.array:
        prompts = list(prompts)
        if not prompts:
            raise ValueError("embed_text_mx requires at least one prompt.")
        return self.embedder.embed_text_mx(prompts).astype(mx.float32)

    def sample_latent_mx(
        self,
        prompt_or_embedding,
        *,
        seed: int,
        guidance_scale: float,
        steps: int,
        sigma_min: float,
        sigma_max: float,
        s_churn: float = 0.0,
        clip_denoised: bool = True,
        progress: bool = False,
        init_x=None,
    ) -> LatentResult:
        timings: dict[str, float] = {}
        if isinstance(prompt_or_embedding, (str, list, tuple)):
            prompts = [prompt_or_embedding] if isinstance(prompt_or_embedding, str) else list(prompt_or_embedding)
            embed_t0 = time.perf_counter()
            embeddings = self.embed_text_mx(prompts)
            timings["embed_seconds"] = time.perf_counter() - embed_t0
        else:
            embeddings = self._to_mx_f32(prompt_or_embedding)

        sample_t0 = time.perf_counter()
        out = karras_sample_heun_mlx(
            diffusion=self.diffusion,
            mlx_model=self.model,
            embeddings=embeddings,
            shape=(int(embeddings.shape[0]), self.text300m_artifact.d_latent),
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            s_churn=s_churn,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            seed=seed,
            init_x=init_x,
            n_ctx=self.text300m_artifact.n_ctx,
            progress=progress,
            strict_torch_math=False,
            return_numpy=False,
        )
        timings["sample_seconds"] = time.perf_counter() - sample_t0
        return LatentResult(
            latent=out["x"].astype(mx.float32),
            pred_xstart=out["pred_xstart"].astype(mx.float32),
            sigmas=out["sigmas"].astype(mx.float32),
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            steps=int(steps),
            timings=timings,
            artifact_info=self.artifact_info,
        )

    def _renderer_bbox(self) -> BoundingBox:
        return BoundingBox(
            min_corner=mx.array(self.renderer.bbox_min).astype(mx.float32),
            max_corner=mx.array(self.renderer.bbox_max).astype(mx.float32),
        )

    def decode_latents_to_meshes_mx(
        self,
        latent,
        *,
        query_batch_size: int = 4096,
        include_vertex_colors: bool = False,
        grid_size: int | None = None,
    ) -> MeshBatchResult:
        decode_t0 = time.perf_counter()
        latent_mx = self._to_mx_f32(latent)
        if latent_mx.ndim == 1:
            latent_mx = mx.expand_dims(latent_mx, axis=0)
        elif latent_mx.ndim != 2:
            raise ValueError(f"latent must be [D] or [B, D], got {latent_mx.shape}")

        params_t0 = time.perf_counter()
        params_mx = self.decoder_params.bottleneck_to_params_mx(latent_mx)
        params_seconds = time.perf_counter() - params_t0

        mesh_t0 = time.perf_counter()
        mesh_batch = self.renderer.extract_meshes(
            params_mx,
            query_batch_size=query_batch_size,
            include_vertex_colors=include_vertex_colors,
            grid_size=grid_size,
        )
        mesh_seconds = time.perf_counter() - mesh_t0
        bbox = self._renderer_bbox()
        meshes: list[MeshResult] = []
        for mesh in mesh_batch.meshes:
            vertex_channels = (
                {name: self._to_mx_f32(values) for name, values in (mesh.vertex_channels or {}).items()}
                if mesh.vertex_channels
                else None
            )
            faces = mesh.faces.astype(mx.int32) if self._is_mx_array(mesh.faces) else mx.array(mesh.faces, dtype=mx.int32)
            meshes.append(
                MeshResult(
                    verts=self._to_mx_f32(mesh.verts),
                    faces=faces,
                    vertex_channels=vertex_channels,
                    bbox=bbox,
                    timings={},
                    artifact_info=self.artifact_info,
                )
            )

        return MeshBatchResult(
            meshes=tuple(meshes),
            bbox=bbox,
            timings={
                "decode_params_seconds": params_seconds,
                "mesh_seconds": mesh_seconds,
                "total_seconds": time.perf_counter() - decode_t0,
            },
            artifact_info=self.artifact_info,
        )

    def decode_latent_to_mesh_mx(
        self,
        latent,
        *,
        query_batch_size: int = 4096,
        include_vertex_colors: bool = False,
        grid_size: int | None = None,
    ) -> MeshResult:
        batch = self.decode_latents_to_meshes_mx(
            latent,
            query_batch_size=query_batch_size,
            include_vertex_colors=include_vertex_colors,
            grid_size=grid_size,
        )
        if len(batch.meshes) != 1:
            raise ValueError(f"Expected a single latent output, but decoded {len(batch.meshes)} meshes. Use decode_latents_to_meshes_mx instead.")
        mesh = batch.meshes[0]
        mesh.timings = dict(batch.timings)
        return mesh
