# Bug Tracker

This file tracks known runtime and conversion bugs. **Note: All P1 and critical stability bugs have been resolved in the consolidated Swift `ShapERuntime` framework.**

## Open

- [ ] P2: Runtime decode API is ambiguous about single-sample vs batched behavior.
  Files:
  [`shap_e/src/mlx_shap_e/runtime_core/pipeline.py`](shap_e/src/mlx_shap_e/runtime_core/pipeline.py)
  [`shap_e/src/cli/run/run_text300m_mlx_pipeline.py`](shap_e/src/cli/run/run_text300m_mlx_pipeline.py)
  Problem:
  The runtime method accepts batch-shaped input, but the CLI explicitly decodes `latent_result.latent[0]`.

## Fixed

- [x] **P1: Production-Grade Swift Port Completion**
  Framework: `ShapERuntime` (Swift/MLX)
  Notes:
  The entire pipeline has been migrated to a high-performance Swift framework. Critical stability fixes for high-resolution meshes (1.3M+ vertices), exact GELU precision matching, and color-aware Marching Cubes are fully implemented and verified.

- [x] Renderer hot path is still not production-clean.
  File:
  [`shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`](shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py)
  Notes:
  Successfully ported to MLX-native (Python).

- [x] Batched mesh decode now keeps samples separate.
  Files:
  [`shap_e/src/mlx_shap_e/runtime_core/pipeline.py`](shap_e/src/mlx_shap_e/runtime_core/pipeline.py)
  [`shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py`](shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py)

- [x] Runtime NumPy-to-MLX conversion bug in `sample_latent_mx()`.
  File:
  [`shap_e/src/mlx_shap_e/runtime_core/pipeline.py`](shap_e/src/mlx_shap_e/runtime_core/pipeline.py)

- [x] Circular import between `runtime_core` and `runtime_artifacts`.
  Files:
  [`shap_e/src/mlx_shap_e/runtime_core/__init__.py`](shap_e/src/mlx_shap_e/runtime_core/__init__.py)
  [`shap_e/src/mlx_shap_e/runtime_artifacts/__init__.py`](shap_e/src/mlx_shap_e/runtime_artifacts/__init__.py)

## Not A Bug

- Production pipeline no longer exposes `--integrator-math`.
  Files:
  [`shap_e/src/cli/run/run_text300m_mlx_pipeline.py`](shap_e/src/cli/run/run_text300m_mlx_pipeline.py)
  [`shap_e/src/cli/run/sample_text300m_karras_mlx.py`](shap_e/src/cli/run/sample_text300m_karras_mlx.py)
  [`shap_e/src/cli/check/check_mlx_karras_loop_parity.py`](shap_e/src/cli/check/check_mlx_karras_loop_parity.py)
  Notes:
  This was an intentional production-runtime change. Parity and fallback behavior still exist in the offline sampling and check scripts.
