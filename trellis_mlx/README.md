# TRELLIS on MLX (port workspace)

Goal: bring Microsoft's [TRELLIS](https://github.com/microsoft/TRELLIS)
image-to-3D pipeline to Apple Silicon following the same discipline as the
Shap-E port — Torch reference → Python MLX parity → exported weight bundle →
Swift runtime — so Mesquette can offer it as a second, higher-quality
generation backend beside Shap-E.

This directory is named `trellis_mlx/` (not `trellis/`) because the
case-insensitive macOS filesystem would collide with the `TRELLIS/`
submodule.

## Sources of truth

| Piece | Location |
|---|---|
| Upstream code | `microsoft/TRELLIS` (MIT), pinned via our fork |
| Fork | `signalnotnoise/TRELLIS` |
| Vendored here | `TRELLIS/` git submodule at `193e0502b5cb3eb5997ec457d5340e95d09b158e` |
| Weights (first target) | `microsoft/TRELLIS-image-large` (MIT), `pipeline.json` rev as downloaded |
| This workspace | `trellis_mlx/` (port code, export scripts, parity evidence) |

Licensing: TRELLIS code and weights are MIT (Microsoft Corporation).
Converted bundles may be redistributed with the MIT notice, same pattern as
`signalNotNoise/shap-e-mlx`. Keep `TRELLIS/LICENSE` attribution in any
published bundle (see repo `NOTICE` conventions).

## Pipeline being ported (image-large)

```
image → DINOv2 ViT-L/14 (dinov2_vitl14_reg) features
      → sparse-structure flow DiT-L  (ss_flow_img_dit_L_16l8_fp16, 16^3 latents)
      → sparse-structure decoder     (ss_dec_conv3d_16l8_fp16) → active voxels
      → SLat flow DiT-L              (slat_flow_img_dit_L_64l8p2_fp16)
      → SLat mesh decoder            (slat_dec_mesh_swin8_B_64l8m256c_fp16)
      → mesh (vertices/faces; vertex colors via mesh decoder output)
```

Samplers: `FlowEulerGuidanceIntervalSampler`, 25 steps, CFG 5.0 on
interval [0.5, 1.0], `rescale_t` 3.0, `sigma_min` 1e-5 (both stages).
Out of scope for the first milestone: Gaussian-splat and radiance-field
decoders, the text pipeline, and training code.

## Hard parts (known unknowns)

- **Sparse tensor ops.** SLat stages run on sparse voxel tensors
  (`TRELLIS/trellis/modules/sparse`, spconv/attention backends upstream).
  MLX has no sparse-conv equivalent; the port must choose
  dense-on-active-voxels reformulations and prove parity per module
  before optimizing.
- **Windowed (Swin) sparse attention** in the mesh decoder.
- **DINOv2 ViT-L/14 with registers** must match upstream preprocessing and
  patch features closely enough for the flow models to behave.
- **Memory.** fp16 checkpoints total several GB across stages; stage-wise
  load/unload (as in ShapERuntime's pipeline switching) is the default plan.

## Working rules (same as Shap-E port)

- Never claim parity without a recorded gate (see `MIGRATION_CHECKLIST.md`).
- Weights and parity dumps stay out of git; bundle format is versioned and
  checksum-verified before any Hub upload.
- `TRELLIS/` changes go to the fork, not here; this repo only moves the pin.
