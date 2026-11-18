import mlx.core as mx
from mlx.utils import tree_unflatten

def to_path(key_str):
    """Convert a dot-separated string to a list of str/int components."""
    parts = key_str.split(".")
    path = []
    for p in parts:
        if p.isdigit():
            path.append(int(p))
        else:
            path.append(p)
    return path

def normalize_sequential_paths(path):
    """Insert 'layers' before integer indices to match MLX nn.Sequential layout."""
    normalized = []
    for p in path:
        if isinstance(p, int):
            if not normalized or normalized[-1] != "layers":
                normalized.append("layers")
        normalized.append(p)
    return normalized

def to_key(path):
    """Convert a list of str/int components back to a dot-separated string."""
    return ".".join(str(p) for p in path)

def map_weights(pt_state_dict):
    flat_weights = []
    
    for key, value in pt_state_dict.items():
        array = mx.array(value.numpy())
        
        # --- A. PROCESSOR BLOCKS (Standard QKV Split) ---
        if "processor.resblocks" in key and "attn.c_qkv" in key:
            q, k, v = mx.split(array, 3, axis=0 if "weight" in key else -1)
            base = key.replace("c_qkv.weight", "").replace("c_qkv.bias", "")
            suffix = "weight" if "weight" in key else "bias"
            
            for name, arr in [("W_q", q), ("W_k", k), ("W_v", v)]:
                path = normalize_sequential_paths(to_path(f"{base}{name}.{suffix}"))
                flat_weights.append((to_key(path), arr))
            continue

        # --- B. ENCODER BLOCKS (OpenAI Split Q + KV) ---
        elif ".attn." in key and ("c_q" in key or "c_kv" in key):
            if "c_q" in key:
                arrs = [("W_q", array)]
            else:
                k, v = mx.split(array, 2, axis=0 if "weight" in key else -1)
                arrs = [("W_k", k), ("W_v", v)]
            
            # Clean up the base path
            base = key.split("attn.")[0] + "attn."
            suffix = "weight" if "weight" in key else "bias"
            
            for name, arr in arrs:
                path = normalize_sequential_paths(to_path(f"{base}{name}.{suffix}"))
                flat_weights.append((to_key(path), arr))
            continue

        # Default path processing
        new_path = to_path(key)
        
        # Handle MLP renaming for Sequential (c_fc -> layers.0, c_proj -> layers.2)
        if "mlp" in new_path and "processor" in new_path:
            # The structure is mlp -> layers -> [index] -> [component]
            # Wait, if we use nn.Sequential, it's mlp.layers.0, mlp.layers.1, mlp.layers.2
            # PT has mlp.c_fc and mlp.c_proj.
            # So the path after loop is: ..., mlp, layers, [?], c_fc, ...
            # Actually, c_fc IS the layer 0 Linear's name in PT? 
            # No, in PT it's often a member. In our Sequential it's just an index.
            if "c_fc" in new_path:
                idx = new_path.index("c_fc")
                new_path[idx] = 0
            elif "c_proj" in new_path:
                idx = new_path.index("c_proj")
                # Attention has out_proj, MLP has layers.2.
                # If 'mlp' is in the path, it's the MLP one.
                new_path[idx] = 2
        
        # Handle Attention output projection renaming
        if "attn" in new_path and "c_proj" in new_path:
            idx = new_path.index("c_proj")
            new_path[idx] = "out_proj"

        new_path = normalize_sequential_paths(new_path)
        flat_weights.append((to_key(new_path), array))

    return tree_unflatten(flat_weights)
