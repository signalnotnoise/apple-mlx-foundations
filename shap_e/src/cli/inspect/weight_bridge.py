import torch
from shap_e.models.download import load_model

# Choose the model you want to inspect: 'transmitter', 'text300M', or 'diffusion'


MODEL_NAME = 'transmitter'
device = torch.device('cpu')

print(f"--- Loading {MODEL_NAME} weights ---")
# This will download the weights if you don't have them
model = load_model(MODEL_NAME, device=device)

# The actual weights are in the state_dict
state_dict = model.state_dict()

print(f"\nFound {len(state_dict)} weight tensors.")
print(f"{'Layer Name':<60} | {'Shape'}")
print("-" * 85)

# Sort and print the first 40 keys to see the hierarchy
keys = state_dict.keys()
for key in keys:
    shape = list(state_dict[key].shape)
    print(f"{key:<60} | {shape}")

print("\n...")