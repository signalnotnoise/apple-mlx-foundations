from pathlib import Path

import mlx.core as mx
import torch
from mlx.utils import tree_flatten, tree_unflatten
from shap_e.models.download import load_model

from mlx_shap_e.models.text300m import Text300MTransformer


def _to_path(key_str: str):
    parts = key_str.split(".")
    path = []
    for part in parts:
        path.append(int(part) if part.isdigit() else part)
    return path


def _normalize_sequential_paths(path):
    normalized = []
    for part in path:
        if isinstance(part, int):
            if not normalized or normalized[-1] != "layers":
                normalized.append("layers")
        normalized.append(part)
    return normalized


def _to_key(path):
    return ".".join(str(part) for part in path)


def map_text300m_weights(pt_state_dict):
    flat_weights = []
    for key, value in pt_state_dict.items():
        new_path = _normalize_sequential_paths(_to_path(key))
        flat_weights.append((_to_key(new_path), mx.array(value.numpy())))
    return tree_unflatten(flat_weights)


def deep_compare_trees(model_dict, weight_dict, path=""):
    for key, value in weight_dict.items():
        current_path = f"{path}.{key}" if path else key
        if key not in model_dict:
            print(f"  [MISSING IN MODEL] {current_path}")
            continue

        model_value = model_dict[key]
        if isinstance(value, dict):
            if not isinstance(model_value, dict):
                print(f"  [TYPE MISMATCH] {current_path}: weight is DICT, model is {type(model_value)}")
            else:
                deep_compare_trees(model_value, value, current_path)
        else:
            if isinstance(model_value, dict):
                print(f"  [TYPE MISMATCH] {current_path}: weight is ARRAY, model is DICT")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "shap_e_model_cache"

    print("Loading text300M checkpoint from local cache...")
    pt_model = load_model("text300M", device=torch.device("cpu"), cache_dir=str(cache_dir))

    print("Initializing MLX text300M model...")
    mlx_model = Text300MTransformer()

    print("Mapping weights...")
    mlx_tree = map_text300m_weights(pt_model.state_dict())

    print("\\n--- DEEP TREE COMPARISON ---")
    model_tree = tree_unflatten(list(tree_flatten(mlx_model.parameters())))
    deep_compare_trees(model_tree, mlx_tree)

    print("\\n--- ATTEMPTING UPDATE ---")
    mlx_model.update(mlx_tree)
    print("Successfully updated MLX model with text300M weights.")

    test_input = mx.random.normal((1, 1024, 1024))
    test_t = mx.array([10.0], dtype=mx.float32)
    test_embeddings = mx.random.normal((1, 768))
    test_output = mlx_model(test_input, test_t, test_embeddings)
    print(f"Sanity forward output shape: {test_output.shape}")


if __name__ == "__main__":
    main()
