import math

import mlx.core as mx
import mlx.nn as nn


def _f32(x: mx.array) -> mx.array:
    if x.dtype != mx.float32:
        return x.astype(mx.float32)
    return x


def timestep_embedding(timesteps: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    """MLX equivalent of shap_e.models.generation.util.timestep_embedding."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = timesteps[:, None] * freqs[None]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)], axis=-1)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = _f32(x)
        return _f32(self.c_proj(nn.gelu(self.c_fc(x))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.c_qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C]
        x = _f32(x)
        bsz, n_ctx, _ = x.shape
        qkv = _f32(self.c_qkv(x))
        qkv = mx.reshape(qkv, (bsz, n_ctx, self.n_heads, 3 * self.head_dim))

        q = qkv[..., : self.head_dim]
        k = qkv[..., self.head_dim : 2 * self.head_dim]
        v = qkv[..., 2 * self.head_dim :]

        # [B, T, H, D] -> [B, H, T, D]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q = _f32(q)
        k = _f32(k)
        v = _f32(v)

        scale = 1.0 / math.sqrt(math.sqrt(self.head_dim))
        weight = _f32(mx.matmul(q * scale, mx.transpose(k * scale, (0, 1, 3, 2))))
        weight = _f32(mx.softmax(weight, axis=-1))

        out = _f32(mx.matmul(weight, v))
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (bsz, n_ctx, self.d_model))
        return _f32(self.c_proj(_f32(out)))


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, d_model)

    def __call__(self, x: mx.array) -> mx.array:
        x = _f32(x)
        h = _f32(self.c_fc(x))
        h = _f32(nn.gelu(h))
        return _f32(self.c_proj(h))


class ResBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, hidden_dim=mlp_hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = _f32(x)
        h1 = _f32(self.ln_1(x))
        x = _f32(x + self.attn(h1))
        h2 = _f32(self.ln_2(x))
        x = _f32(x + self.mlp(h2))
        return _f32(x)


class Backbone(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResBlock(d_model=d_model, n_heads=n_heads, mlp_hidden_dim=mlp_hidden_dim) for _ in range(n_layers)]
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.resblocks(x)


class WrappedText300M(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        mlp_hidden_dim: int = 4096,
        input_channels: int = 1024,
        output_channels: int = 2048,
        clip_feature_dim: int = 768,
        token_cond: bool = True,
        time_token_cond: bool = True,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_cond = token_cond
        self.time_token_cond = time_token_cond
        self.use_pos_emb = use_pos_emb

        self.input_proj = nn.Linear(input_channels, d_model)
        self.clip_embed = nn.Linear(clip_feature_dim, d_model)
        self.backbone = Backbone(d_model=d_model, n_layers=n_layers, n_heads=n_heads, mlp_hidden_dim=mlp_hidden_dim)
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_channels)
        self.pos_emb = mx.zeros((1024, d_model))
        self.time_embed = TimeEmbedding(d_model=d_model, hidden_dim=mlp_hidden_dim)

    def _forward_with_cond(self, x_nct: mx.array, cond_as_token):
        # NCT -> NTC
        h = _f32(self.input_proj(mx.transpose(_f32(x_nct), (0, 2, 1))))

        for emb, as_token in cond_as_token:
            if not as_token:
                h = _f32(h + _f32(emb)[:, None, :])

        if self.use_pos_emb:
            h = _f32(h + _f32(self.pos_emb[: h.shape[1]])[None, :, :])

        extra_tokens = [_f32(emb)[:, None, :] for emb, as_token in cond_as_token if as_token]
        if extra_tokens:
            h = mx.concatenate(extra_tokens + [h], axis=1)
            h = _f32(h)

        h = _f32(self.ln_pre(_f32(h)))
        h = _f32(self.backbone(_f32(h)))
        h = _f32(self.ln_post(_f32(h)))

        if extra_tokens:
            n_extra = sum(tok.shape[1] for tok in extra_tokens)
            h = h[:, n_extra:, :]

        h = _f32(self.output_proj(_f32(h)))
        # NTC -> NCT
        return _f32(mx.transpose(h, (0, 2, 1)))

    def __call__(self, x: mx.array, t: mx.array, embeddings: mx.array) -> mx.array:
        t_embed = _f32(self.time_embed(timestep_embedding(_f32(t), self.d_model)))
        clip_out = _f32(math.sqrt(embeddings.shape[-1]) * _f32(embeddings))
        clip_embed = _f32(self.clip_embed(clip_out))

        cond = [(clip_embed, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, cond)


class Text300MTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = WrappedText300M()

    def __call__(self, x: mx.array, t: mx.array, embeddings: mx.array) -> mx.array:
        return self.wrapped(x, t, embeddings)
