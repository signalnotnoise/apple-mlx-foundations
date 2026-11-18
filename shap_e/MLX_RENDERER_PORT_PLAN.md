# MLX-Native Renderer Port Plan

> Historical planning note. The file paths below describe the layout used
> while the port was being planned. Current implementations live in
> `shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`,
> `shap_e/src/mlx_shap_e/models/rendering/marching_cubes.py`,
> `shap_e/swift/Sources/ShapERuntime/STFMeshRenderer.swift`, and
> `shap_e/swift/Sources/ShapERuntime/MarchingCubes.swift`. See
> [`docs/FILE-STRUCTURE.md`](../docs/FILE-STRUCTURE.md) for the current map.

## Goal
Remove dependency on PyTorch renderer execution for mesh extraction by introducing an MLX-native STF path.

## Scope Split
1. Minimal geometry path (current phase)
- Input: decoder latent (`[D]`) and decoder weights.
- Output: mesh geometry (`verts`, `faces`) from STF signed-distance field.
- Excludes: camera rasterization, shading, texture channels.

2. Full renderer parity (future phase)
- Add texture channel evaluation and colorized mesh output.
- Add view rendering parity for images/transmittance.

## Current Implementation Status (Started)
Completed in this repo:
- Added `shap_e/src/mlx_stf_mesh_renderer.py`.
- Implemented minimal NeRSTF forward path for signed distance:
  - NeRF positional encoding.
  - Meta/dynamic layers `nerstf.mlp.0..3` from latent-projected weights.
  - Static layers `renderer.nerstf.mlp.4..6`.
  - `swish` hidden activations and `tanh` SDF output.
- Implemented dense bbox query + negative border padding + marching cubes extraction.
- Added decode integration flag:
  - `--renderer-backend torch|mlx` in
    - `shap_e/src/decode_text300m_latent_to_mesh.py`
    - `shap_e/src/sample_text300m_karras_mlx.py`
- Added `--query-batch-size` for MLX SDF grid eval.

## Near-Term Tasks
1. Parity checks
- Compare torch vs mlx geometry outputs on fixed latent:
  - vertex count
  - face count
  - bbox overlap
  - Chamfer distance (optional)

2. Stability/perf
- Batch-size and memory tuning for 128^3 evaluation.
- Add optional lower/higher grid size for debug/perf.

3. Remove torch extraction bridge
- Replace `from_torch_decoder(...)` static-weight bootstrap with direct non-torch weight loading artifact (e.g., NPZ).
- Then MLX renderer can run without instantiating torch decoder objects.

## Risks
- Exact mesh parity may vary due to marching-cubes implementation differences.
- Current fallback mesh extraction path may produce occupancy-style surfaces if `skimage` is unavailable.
- Texture/color outputs are not yet ported.

## How To Exercise
Decode with MLX mesh backend:

```bash
python shap_e/src/decode_text300m_latent_to_mesh.py \
  --latent-npz shap_e/outputs/trajectory_a-red-chair.npz \
  --renderer-backend mlx \
  --query-batch-size 4096
```
