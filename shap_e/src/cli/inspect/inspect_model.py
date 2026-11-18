from mlx_shape_data_structures import ShapETransformer
from mlx.utils import tree_flatten

def main():
    model = ShapETransformer(d_model=1024, n_layers=12, n_heads=16)
    params = dict(tree_flatten(model.parameters()))
    
    print("\n--- MODEL TOP-LEVEL PARAMETERS ---")
    for k in params.keys():
        if "." not in k:
            print(f"Top-level: {k}")
    
    if "output_tokens" in params:
        print("output_tokens FOUND in parameters!")
    else:
        print("output_tokens NOT FOUND in parameters!")

if __name__ == "__main__":
    main()
