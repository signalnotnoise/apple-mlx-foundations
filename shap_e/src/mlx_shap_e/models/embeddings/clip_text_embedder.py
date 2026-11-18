from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import zipfile

import mlx.core as mx
import numpy as np

from mlx_shap_e.models.embeddings.clip_tokenizer import ClipSimpleTokenizer, default_clip_bpe_path


@dataclass
class MlxClipTextArtifact:
    context_length: int
    transformer_width: int
    transformer_heads: int
    transformer_layers: int
    state: dict[str, np.ndarray]

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        *,
        regenerate_fn=None,
    ) -> "MlxClipTextArtifact":
        path = Path(artifact_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"MLX CLIP text artifact not found: {path}. "
                "Generate with: python shap_e/src/export_mlx_clip_text_artifact.py"
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                artifact_type = str(data["artifact_type"].item())
                if artifact_type != "mlx_clip_text_vit_l14_v1":
                    raise ValueError(f"Unsupported CLIP artifact type: {artifact_type}")
                state = {k: np.asarray(v) for k, v in data.items() if k not in {"artifact_type"}}
        except zipfile.BadZipFile:
            if regenerate_fn is None:
                raise
            regenerate_fn()
            with np.load(path, allow_pickle=False) as data:
                artifact_type = str(data["artifact_type"].item())
                if artifact_type != "mlx_clip_text_vit_l14_v1":
                    raise ValueError(f"Unsupported CLIP artifact type: {artifact_type}")
                state = {k: np.asarray(v) for k, v in data.items() if k not in {"artifact_type"}}
        return cls(
            context_length=int(state.pop("meta_context_length").item()),
            transformer_width=int(state.pop("meta_transformer_width").item()),
            transformer_heads=int(state.pop("meta_transformer_heads").item()),
            transformer_layers=int(state.pop("meta_transformer_layers").item()),
            state=state,
        )


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    y = mx.matmul(x, mx.transpose(weight, (1, 0)))
    if bias is not None:
        y = y + bias
    return y


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    x_norm = (x - mean) * mx.rsqrt(var + eps)
    return x_norm * weight + bias


class MlxClipTextEmbedder:
    def __init__(
        self,
        artifact: MlxClipTextArtifact,
        *,
        tokenizer_bpe_path: str | Path | None = None,
    ):
        bpe_path = Path(tokenizer_bpe_path).expanduser().resolve() if tokenizer_bpe_path else default_clip_bpe_path()
        if not bpe_path.exists():
            raise FileNotFoundError(f"CLIP tokenizer merges file not found: {bpe_path}")
        self.artifact = artifact
        self.tokenizer = ClipSimpleTokenizer(bpe_path=bpe_path)
        self.mx_state = {k: mx.array(v.astype(np.float32)) for k, v in artifact.state.items()}
        self._attn_mask_cache: dict[int, mx.array] = {}

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        tokenizer_bpe_path: str | Path | None = None,
        regenerate_fn=None,
    ) -> "MlxClipTextEmbedder":
        return cls(
            MlxClipTextArtifact.load(artifact_path, regenerate_fn=regenerate_fn),
            tokenizer_bpe_path=tokenizer_bpe_path,
        )

    def embed_text_mx(self, prompts: list[str]) -> mx.array:
        tokens_np = self.tokenizer.tokenize(prompts, context_length=self.artifact.context_length)
        tokens = mx.array(tokens_np, dtype=mx.int32)

        state = self.mx_state
        tok_w = state["token_embedding.weight"]
        pos = state["positional_embedding"]
        x = tok_w[tokens] + pos[None, :, :]

        n_heads = self.artifact.transformer_heads
        n_layers = self.artifact.transformer_layers
        d_model = self.artifact.transformer_width
        head_dim = d_model // n_heads

        seq_len = int(x.shape[1])
        attn_mask = self._causal_attn_mask(seq_len)

        for i in range(n_layers):
            prefix = f"transformer.resblocks.{i}"
            ln1 = _layer_norm(
                x,
                state[f"{prefix}.ln_1.weight"],
                state[f"{prefix}.ln_1.bias"],
            )
            qkv = _linear(
                ln1,
                state[f"{prefix}.attn.in_proj_weight"],
                state[f"{prefix}.attn.in_proj_bias"],
            )
            q, k, v = mx.split(qkv, 3, axis=-1)
            bsz = int(q.shape[0])
            q = mx.transpose(mx.reshape(q, (bsz, seq_len, n_heads, head_dim)), (0, 2, 1, 3))
            k = mx.transpose(mx.reshape(k, (bsz, seq_len, n_heads, head_dim)), (0, 2, 1, 3))
            v = mx.transpose(mx.reshape(v, (bsz, seq_len, n_heads, head_dim)), (0, 2, 1, 3))

            scale = 1.0 / math.sqrt(float(head_dim))
            attn_scores = mx.matmul(q * scale, mx.transpose(k, (0, 1, 3, 2))) + attn_mask
            attn_probs = mx.softmax(attn_scores, axis=-1)
            attn_out = mx.matmul(attn_probs, v)
            attn_out = mx.reshape(mx.transpose(attn_out, (0, 2, 1, 3)), (bsz, seq_len, d_model))
            attn_out = _linear(
                attn_out,
                state[f"{prefix}.attn.out_proj.weight"],
                state[f"{prefix}.attn.out_proj.bias"],
            )
            x = x + attn_out

            ln2 = _layer_norm(
                x,
                state[f"{prefix}.ln_2.weight"],
                state[f"{prefix}.ln_2.bias"],
            )
            mlp_h = _linear(
                ln2,
                state[f"{prefix}.mlp.c_fc.weight"],
                state[f"{prefix}.mlp.c_fc.bias"],
            )
            mlp_h = mlp_h * mx.sigmoid(mx.array(1.702, dtype=mlp_h.dtype) * mlp_h)
            mlp_out = _linear(
                mlp_h,
                state[f"{prefix}.mlp.c_proj.weight"],
                state[f"{prefix}.mlp.c_proj.bias"],
            )
            x = x + mlp_out

        x = _layer_norm(x, state["ln_final.weight"], state["ln_final.bias"])
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        eot_indices = self._select_pool_indices(tokens, eot_token=eot_token)
        batch_indices = mx.arange(x.shape[0], dtype=mx.int32)
        gathered = x[batch_indices, eot_indices, :]
        text_proj = state["text_projection"]
        emb = mx.matmul(gathered, text_proj)

        denom = mx.sqrt(mx.sum(emb * emb, axis=-1, keepdims=True))
        emb = emb / mx.maximum(denom, mx.array(1e-12, dtype=emb.dtype))
        return emb.astype(mx.float32)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        # Compatibility wrapper for existing NumPy-based call sites.
        emb = self.embed_text_mx(prompts)
        return np.asarray(emb, dtype=np.float32)

    def _causal_attn_mask(self, seq_len: int) -> mx.array:
        cached = self._attn_mask_cache.get(seq_len)
        if cached is not None:
            return cached
        row = mx.arange(seq_len)[:, None]
        col = mx.arange(seq_len)[None, :]
        neg_inf = mx.array(-1e9, dtype=mx.float32)
        zeros = mx.array(0.0, dtype=mx.float32)
        mask = mx.where(col > row, neg_inf, zeros)[None, None, :, :]
        self._attn_mask_cache[seq_len] = mask
        return mask

    @staticmethod
    def _select_pool_indices(tokens: mx.array, *, eot_token: int) -> mx.array:
        seq_len = int(tokens.shape[1])
        positions = mx.arange(seq_len)[None, :]
        eot_mask = tokens == mx.array(eot_token, dtype=tokens.dtype)
        has_eot = mx.any(eot_mask, axis=-1)
        eot_positions = mx.where(eot_mask, positions, mx.array(-1, dtype=positions.dtype))
        last_eot = mx.max(eot_positions, axis=-1)
        nonzero_positions = mx.where(tokens != 0, positions, mx.array(-1, dtype=positions.dtype))
        last_nonzero = mx.max(nonzero_positions, axis=-1)
        return mx.where(has_eot, last_eot, last_nonzero).astype(mx.int32)
