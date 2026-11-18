import argparse
import os
import torch
import mlx.core as mx
from mlx_shape_data_structures import ShapETransformer
from shap_e.models.download import load_model, load_config
from safetensors.torch import save_file
from mlx_weight_remapper import map_weights
from mlx.utils import tree_flatten
import sys

# Add the shap-e directory to sys.path so we can import shap_e
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shap-e")))

# from shap_e.models.download import MODEL_PATHS, load_checkpoint, load_config

# def convert_weights(checkpoint_name, output_path):
    # print(f"Loading checkpoint: {checkpoint_name}...")
    # device = torch.device("cpu")
    # state_dict = load_checkpoint(checkpoint_name, device=device)
    
    # # MLX expects weights in a specific format usually, but for now we just 
    # # convert torch tensors to numpy then to mlx or just use safetensors.
    # # Actually, mlx.core.save_safetensors is compatible with safetensors.
    
    # converted_weights = {}
    # for k, v in state_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         # Convert to numpy
    #         val = v.detach().cpu().numpy()
            
    #         # Print info for debugging
    #         # print(f"Key: {k}, Shape: {val.shape}, Dtype: {val.dtype}")
            
    #         # Map problematic dtypes if necessary
    #         if val.dtype == "float64":
    #             val = val.astype("float32")
    #         elif val.dtype == "int64":
    #             val = val.astype("int32")
            
    #         converted_weights[k] = val
    #     else:
    #         print(f"Skipping non-tensor key: {k}")

    # output_file = os.path.join(output_path, f"{checkpoint_name}.safetensors")
    # print(f"Saving {len(converted_weights)} tensors to {output_file}...")
    
    # # Use torch's safetensors saver instead of MLX's for weight conversion
    # # as it's more robust to PyTorch-originated tensors.
    # from safetensors.torch import save_file
    
    # # Convert numpy back to torch for saving with torch's safetensors
    # torch_weights = {k: torch.from_numpy(v) for k, v in converted_weights.items()}
    
    # try:
    #     save_file(torch_weights, output_file)
    #     print(f"Successfully converted {checkpoint_name}")
    # except Exception as e:
    #     print(f"Failed to save {checkpoint_name} with torch: {e}")
    #     raise e

def deep_compare_trees(model_dict, weight_dict, path=""):
    """Recursively find mismatches between model params and weight tree."""
    for k, v in weight_dict.items():
        curr_path = f"{path}.{k}" if path else k
        if k not in model_dict:
            print(f"  [MISSING IN MODEL] {curr_path}")
            continue
            
        model_v = model_dict[k]
        if isinstance(v, dict):
            if not isinstance(model_v, dict):
                print(f"  [TYPE MISMATCH] {curr_path}: weight is DICT, model is {type(model_v)}")
            else:
                deep_compare_trees(model_v, v, curr_path)
        else:
            if isinstance(model_v, dict):
                print(f"  [TYPE MISMATCH] {curr_path}: weight is ARRAY, model is DICT")

def main():
    # 1. Load Original PyTorch Model
    print("Fetching PyTorch weights...")
    # text_model = load_model('text300M', device=torch.device('cpu'))
    pt_model = load_model('transmitter', device=torch.device('cpu'))
    
    # 2. Initialize MLX Model
    print("Initializing MLX model...")
    
    model = ShapETransformer(d_model=1024, n_layers=12, n_heads=16)
    
    # 3. Migrate Weightså
    print("Migrating weights...")
    mlx_tree = map_weights(pt_model.state_dict())
    
    # 4. Diagnostics
    print("\n--- DEEP TREE COMPARISON ---")
    # Convert model parameters to a nested dict for easier comparison
    from mlx.utils import tree_unflatten
    model_tree = tree_unflatten(list(tree_flatten(model.parameters())))
    deep_compare_trees(model_tree, mlx_tree)

    print("\n--- ATTEMPTING UPDATE ---")
    try:
        model.update(mlx_tree)
        print("Successfully converted Transmitter to MLX!")
    except Exception as e:
        print(f"\nUPATE FAILED: {e}")
        raise e
    
    # Test with a dummy input
    test_input = mx.random.normal((1, 100, 1024))
    output = model(test_input)
    print(f"Test output shape: {output.shape}")

if __name__ == "__main__":
    main()
