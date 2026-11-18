"""Legacy standalone helper for loading a PyTorch state dict into MLX."""

import mlx.core as mx
from mlx.utils import tree_unflatten


def load_pytorch_to_mlx(mlx_model, pt_state_dict):
    """Map selected PyTorch attention/MLP keys and update an MLX model."""
    mlx_weights = {}
    
    for key, value in pt_state_dict.items():
        # 1. Convert to MLX array
        array = mx.array(value.numpy())
        
        # 2. Handle the c_attn split (OpenAI specific)
        if "c_attn" in key:
            # Splits the weight/bias into 3 equal parts for Q, K, and V
            q, k, v = mx.split(array, 3, axis=0 if "weight" in key else -1)
            
            base_key = key.replace("c_attn", "attn")
            mlx_weights[f"{base_key}.W_q"] = q
            mlx_weights[f"{base_key}.W_k"] = k
            mlx_weights[f"{base_key}.W_v"] = v
        
        # 3. Standard Linear mapping (c_fc and c_proj)
        elif "c_fc" in key:
            mlx_weights[key.replace("c_fc", "mlp.layers.0")] = array
        elif "c_proj" in key:
            # In our ShapEBlock, the second linear is index 2
            if "mlp" in key:
                mlx_weights[key.replace("c_proj", "mlp.layers.2")] = array
            else:
                mlx_weights[key.replace("c_proj", "attn.out_proj")] = array
        else:
            mlx_weights[key] = array

    # Update the model
    mlx_model.update(tree_unflatten(list(mlx_weights.items())))
    print("Weights successfully migrated!")