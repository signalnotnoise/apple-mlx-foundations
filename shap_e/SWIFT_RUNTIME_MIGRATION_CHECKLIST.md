# Swift Runtime Migration Checklist

> Historical migration record. Current paths, packages, and runtime-bundle
> layout are maintained in [`docs/FILE-STRUCTURE.md`](../docs/FILE-STRUCTURE.md)
> and [`knowledge_graph/README.md`](../knowledge_graph/README.md); where this
> checklist and current code disagree, current code wins.

Goal: make inference-time code path free of NumPy/PyTorch so the same math can be implemented in Swift + MLX.

## Scope

- Runtime path: prompt -> text embedding -> diffusion sampling -> latent -> mesh decode.
- Out of scope: conversion/export/check scripts that run offline in Python.

## Current status

### Done in Python MLX runtime

- `shap_e/src/mlx_shap_e/models/text300m.py`
  - Core model forward is MLX-native.
- `shap_e/src/mlx_shap_e/models/embeddings/clip_text_embedder.py`
  - Added `embed_text_mx(...)` returning `mx.array` for runtime use.
  - Existing `embed_text(...)` now acts as a NumPy compatibility wrapper.
- `shap_e/src/mlx_shap_e/models/decoder_params.py`
  - `bottleneck_to_params_mx(...)` exists for runtime latent -> decoder params.
- `shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`
  - MLX SDF eval, batched field sampling, and mesh extraction path already exist.
- `shap_e/src/mlx_shap_e/runtime_core/pipeline.py`
  - End-to-end Python MLX runtime exists: prompt -> embedding -> sample -> latent -> mesh.

### Done in Swift runtime bootstrap

- `shap_e/swift/Package.swift`
  - Added an initial standalone Swift package for the Shap-E runtime port.
- `shap_e/swift/Sources/ShapERuntime/RuntimeManifest.swift`
  - Swift manifest models now load the exported runtime bundle contract.
- `shap_e/swift/Sources/ShapERuntime/TensorBlobLoader.swift`
  - Raw tensor blob loading with dtype/shape validation exists in Swift.
- `shap_e/swift/Sources/ShapERuntime/RuntimeArtifactBundle.swift`
  - Swift runtime bundle loader can read family manifests, family versions, and tensor blobs.
- `shap_e/swift/Sources/ShapERuntime/RuntimeCoreTypes.swift`
  - Ported shared runtime result/version structs used by the Swift runtime surface.
- `shap_e/swift/Sources/ShapERuntime/Models/embeddings/ClipTextArtifact.swift`
  - CLIP artifact metadata/state can now be loaded from the runtime bundle in Swift.
- `shap_e/swift/Sources/ShapERuntime/Models/embeddings/ClipSimpleTokenizer.swift`
  - CLIP BPE tokenizer is ported in Swift, including gzip vocab loading, BPE merges, padding, truncation, and Unicode normalization close to the Python runtime path.
- `shap_e/swift/Sources/ShapERuntime/Text300MTransformer.swift`
  - Full Text300M diffusion transformer ported to native MLXArray ops (matmul, softmax, layer norm, QuickGELU). All scalar [Float] helpers removed.
- `shap_e/swift/Sources/ShapERuntime/DiffusionStep.swift`
  - split_vector_forward, _extract_into, and p_mean_variance (learned-range) ported from diffusion_step.py.
- `shap_e/swift/Sources/ShapERuntime/KarrasSampler.swift`
  - Karras sigma schedule, sigma→timestep interpolation, Gaussian-to-Karras denoiser, and Heun ODE loop with optional CFG ported from the MLX path of karras_sampler.py.
- `shap_e/swift/Sources/ShapERuntime/DecoderParamsProjector.swift`
  - `bottleneck_to_params_mx` ported: latent unwarp (identity / tan2), einsum projection, per-group layer norm, bias add, and reshape to `[B, *param_shape]`.
- `shap_e/swift/Tests/ShapERuntimeTests/*`
  - 36 tests executed at the 2026-11-17 follow-up audit (33 passing, 3 skipped). Covers manifest loading, CLIP embedder, Text300M transformer, DiffusionStep math, DecoderParamsProjector, image preprocessing orientation, and artifact validation.

### Still needed for Swift / Vision Pro runtime

- `shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`
  - ✅ Ported MLX renderer math to Swift (`STFMeshRenderer.swift`, including `extractSDFFields` and `volumeQueryPoints`).
- `shap_e/src/mlx_shap_e/models/rendering/marching_cubes.py`
  - ✅ Ported marching cubes native equivalent (`MarchingCubes.swift`).
- End-to-end assembly
  - ✅ Recreated the full Python pipeline surface in `ShapERuntime.swift`: prompt → CLIP → latent → mesh.

### Python tooling (can remain NumPy)

- `shap_e/src/cli/export/*`
- `shap_e/src/mlx_shap_e/models/conversions/*`
- `shap_e/src/cli/check/*`
- Artifact readers/writers (`np.load`, `np.savez_compressed`) used offline.

## Swift-first port order

1. Artifact loading + runtime contract
   - Port the runtime bundle format first so every later stage reads the same exported tensors.
   - Status: foundation completed in `shap_e/swift`.
   - Python source:
     - `shap_e/src/mlx_shap_e/runtime_artifacts/bundle.py`
     - `shap_e/src/mlx_shap_e/runtime_artifacts/manifest.py`
     - `shap_e/src/mlx_shap_e/runtime_core/types.py`
2. Tokenization + CLIP embedding
   - Port tokenizer and CLIP text embedder before touching diffusion.
   - Status: ✅ DONE — tokenizer parity + CLIP embedder both passing.
   - Python source:
     - `shap_e/src/mlx_shap_e/models/embeddings/clip_tokenizer.py`
     - `shap_e/src/mlx_shap_e/models/embeddings/clip_text_embedder.py`
3. Text300M model forward
   - Port the transformer that predicts `[eps, var]`.
   - Status: ✅ DONE — full MLXArray port, 3/3 tests passing.
   - Python source:
     - `shap_e/src/mlx_shap_e/models/text300m.py`
     - `shap_e/src/mlx_shap_e/models/artifacts/text300m_artifact.py`
4. Diffusion helper math
   - Port timestep embedding reshape/layout assumptions and learned-range PMV logic.
   - Status: ✅ DONE — split_vector_forward, extractInto, pMeanVariance, 6/6 tests passing.
   - Python source:
     - `shap_e/src/mlx_shap_e/models/diffusion/diffusion_step.py`
5. Karras sampler
   - Port only the MLX production branch, not the Python Torch parity branch.
   - Status: ✅ DONE — sigma schedule, sigma→t interpolation, Heun loop, CFG, channel unscaling.
   - Python source:
     - `shap_e/src/mlx_shap_e/models/diffusion/karras_sampler.py`
6. Latent → decoder params projection
   - Port the existing MLX projection path.
   - Status: ✅ DONE — unwarp (identity/tan2), einsum projection, layer norm, bias, reshape; 6/6 tests passing.
   - Python source:
      - `shap_e/src/mlx_shap_e/models/decoder_params.py`
7. Renderer SDF/color eval
   - Port NeRSTF positional encoding, dynamic meta layers, static layers, and field sampling.
   - Status: ✅ DONE
   - Python source:
     - `shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`
8. Marching cubes
   - Port exact edge indexing + interpolation, or replace with Swift-native equivalent.
   - Status: ✅ DONE
   - Python source:
     - `shap_e/src/mlx_shap_e/models/rendering/marching_cubes.py`
9. End-to-end runtime assembly
   - Recreate the Python runtime surface in Swift: prompt → token ids → CLIP → latent → mesh.
   - Status: ✅ DONE
   - Python source:
     - `shap_e/src/mlx_shap_e/runtime_core/pipeline.py`

## File-by-file mapping

### Loader / artifacts

- Python: `shap_e/src/mlx_shap_e/runtime_artifacts/bundle.py`
- Swift:
  - `RuntimeArtifactBundle.swift`
  - `RuntimeManifest.swift`
  - `TensorBlobLoader.swift`

### Tokenizer / text embedding

- Python: `shap_e/src/mlx_shap_e/models/embeddings/clip_tokenizer.py`
- Swift:
  - `ClipSimpleTokenizer.swift`
- Python: `shap_e/src/mlx_shap_e/models/embeddings/clip_text_embedder.py`
- Swift:
  - `ClipTextEmbedder.swift`
  - `ClipTextArtifact.swift`

### Diffusion model

- Python: `shap_e/src/mlx_shap_e/models/text300m.py`
- Swift: ✅ `Text300MTransformer.swift` (includes TimestepEmbedding — WrappedText300M is not needed)
- Python: `shap_e/src/mlx_shap_e/models/artifacts/text300m_artifact.py`
- Swift: ✅ `Text300MArtifact.swift`

### Diffusion sampling

- Python: `shap_e/src/mlx_shap_e/models/diffusion/diffusion_step.py`
- Swift: ✅ `DiffusionStep.swift`
- Python: `shap_e/src/mlx_shap_e/models/diffusion/karras_sampler.py`
- Swift: ✅ `KarrasSampler.swift` (includes GaussianToKarrasDenoiser as a class within the file)

### Decoder params

- Python: `shap_e/src/mlx_shap_e/models/decoder_params.py`
- Swift: ✅
  - `DecoderParamsProjector.swift`
  - `RuntimeArtifactTypes.swift` (`DecoderParamsArtifact`)

### Renderer / meshing

- Python: `shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`
- Swift: ✅
  - `STFMeshRenderer.swift`
  - `STFMeshRenderer.swift` field helpers `extractSDFFields` and `volumeQueryPoints`
- Python: `shap_e/src/mlx_shap_e/models/rendering/marching_cubes.py`
- Swift: ✅
  - `MarchingCubes.swift`

### End-to-end runtime

- Python: `shap_e/src/mlx_shap_e/runtime_core/pipeline.py`
- Swift: ✅
  - `ShapERuntime.swift`

## Validation gates

Three layers — see `shap_e/parity/README.md` for full commands.

### L0 — Torch ↔ Python MLX (existing `cli/check/*`)

Upstream when re-exporting weights. Report-only metrics. Unchanged.

### L1 — MLX self-golden

1. Tokenizer / CLIP / sampler heads / mesh counts
   - Status: ✅ PASSING via `check_mlx_runtime_golden.py` + fixture `tests/fixtures/runtime_golden_a_red_chair_steps2.json`
2. Decoder params (Torch↔MLX)
   - Status: use `check_mlx_decoder_params_parity.py` (report-only)
3. Mesh renderer structural
   - Status: use `check_mlx_mesh_renderer_parity.py`

### L2 — Python MLX ↔ Swift full tensors (new)

- Status: ✅ PASSING smoke (`steps=2`, `grid=16`, prompt `a red chair`, seed 7)
- One-shot: `./shap_e/scripts/run_parity.sh`
- Stages: tokens (exact), CLIP (exact), sigmas (exact), latent / pred_xstart / decoder params / SDF (corr≥0.999), CPU mesh verts corr/mean + exact faces (Metal: counts + bbox)
- Fixes landed during harness bring-up:
  - Shared `python_init_x.npy` (NumPy RNG vs MLXRandom)
  - CFG uncond zeros in parity CLI (match Python Karras)
  - Integer lattice for volume query points in Swift (`volumeQueryPoints`)

Python / Swift validation entry points:

- `shap_e/src/cli/check/check_mlx_runtime_golden.py` (L1)
- `shap_e/src/cli/check/dump_mlx_swift_parity.py` (L2 dump)
- `shap_e/src/cli/check/compare_mlx_swift_parity.py` (L2 assert)
- `shap_e/swift` product `ShapECLI --parity`
- `shap_e/scripts/run_parity.sh`
- `shap_e/src/mlx_shap_e/runtime_core/golden.py`
- `shap_e/src/mlx_shap_e/runtime_core/parity_tensors.py`

## Definition of done

- No NumPy/PyTorch dependency in selected production runtime path.
- Swift app can reproduce Python MLX latent outputs within agreed tolerance.
- Mesh output path in Swift uses equivalent decoder/renderer math and passes visual + numeric checks.
- Vision Pro app consumes exported runtime artifacts directly and does not depend on Python at runtime.
