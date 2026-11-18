import mlx.core as mx
import mlx.nn as nn

class UnkTokenLinear(nn.Linear):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__(input_dims, output_dims, bias)
        self.unk_token = mx.zeros((1, output_dims))

# Removed ShapEEncoder in favor of ShapEEncoderUnit below

class PointConv(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp_convs = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model)
        )

class ShapEAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # Matches c_q
        self.W_q = nn.Linear(d_model, d_model)
        # Matches c_kv (we split this in the remapper)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        # Matches c_proj
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        return self.out_proj(self.W_q(x) + self.W_k(x) + self.W_v(x))   

class Resblock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = ShapEAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(d_model, 4096)
        self.mlp.c_proj = nn.Linear(4096, d_model)
        self.ln_3 = nn.LayerNorm(d_model)
    def __call__(self, x, mask=None):
        h = self.ln_1(x)
        x = x + self.attn(h, mask)
        x = x + self.mlp.c_proj(mx.gelu(self.mlp.c_fc(self.ln_2(x))))
        return self.ln_3(x)
        


class Renderer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.nerstf = NeRSTF(d_model)
        self.void = Void()

class NeRSTF(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(*[nn.Linear(d_model, d_model) for _ in range(7)])

class Void(nn.Module):
    def __init__(self):
        super().__init__()
        self.background = mx.zeros((1,)) # Placeholder



class ParamsProj(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.projections = NeRFProjection(d_model, out_dim)

class NeRFProjection(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        # We define the 4 heads as members of this class
        # This matches the key: projections.head_0...
        self.nerstf__mlp__0__weight = ProjectionHead(d_model, out_dim)
        self.nerstf__mlp__1__weight = ProjectionHead(d_model, out_dim)
        self.nerstf__mlp__2__weight = ProjectionHead(d_model, out_dim)
        self.nerstf__mlp__3__weight = ProjectionHead(d_model, out_dim)



class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        # These names must match the middle part of your keys
        self.proj = nn.Linear(d_model, out_dim)
        self.norm = nn.LayerNorm(out_dim)

class ShapEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        # Use ShapEAttention so parameter names align with PT mapping (W_q/W_k/W_v/out_proj)
        self.attn = ShapEAttention(d_model, n_heads)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(*[
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        ])

    # def __call__(self, x, mask=None):
    #     # Attention path with residual
    #     h = self.ln_1(x)
    #     x = x + self.attn(h, h, h, mask)
        
    #     # MLP path with residual
    #     x = x + self.mlp(self.ln_2(x))
    #     return x

class ShapEProcessor(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[ShapEBlock(d_model, n_heads) for _ in range(n_layers)])

class ShapEEncoderUnit(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.resblocks = nn.Sequential(*[Resblock(d_model, n_heads) for _ in range(1)])

class TransmitterEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        self.ln_pre = nn.LayerNorm(d_model)
        self.ln_post = nn.LayerNorm(d_model)
        
        # MLX tracks mx.array attributes as parameters
        self.output_tokens = mx.zeros((1, 1024))
        self.output_proj = nn.Linear(d_model, 1024)
        
        self.mv_pcl_embed = UnkTokenLinear(3, d_model)
        self.patch_emb = nn.Linear(3, d_model)
        self.point_conv = PointConv(d_model)
        
        self.processor = ShapEProcessor(d_model, n_layers, n_heads)
        self.encoder = nn.Sequential(*[ShapEEncoderUnit(d_model, n_heads) for _ in range(2)])
        self.params_proj = ParamsProj(d_model, 1024)

    def __call__(self, x):
        return x # Placeholder



class ShapETransformer(nn.Module):
    def __init__(self, d_model: int=1024, n_layers: int = 12, n_heads: int = 16):
        super().__init__()
        self.encoder = TransmitterEncoder(d_model, n_layers, n_heads)
        self.renderer = Renderer(d_model)

    def __call__(self, x):
        return self.encoder(x)
