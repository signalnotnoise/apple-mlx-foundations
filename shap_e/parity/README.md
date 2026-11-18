# Shap-E parity (Python MLX ↔ Swift)

Three layers of numerical checks. **Do not replace L0/L1** — extend them.

| Layer | What | Entry points | Grade |
|-------|------|--------------|-------|
| **L0** | OpenAI Torch ↔ Python MLX | `shap_e/src/cli/check/check_*.py` | Report metrics (human) |
| **L1** | MLX self-golden (summary JSON) | `check_mlx_runtime_golden.py`, `tests/fixtures/runtime_golden_*.json` | Assert `atol=1e-4` |
| **L2** | Full tensors Python MLX ↔ Swift | `dump_mlx_swift_parity` → `ShapECLI --parity` → `compare_mlx_swift_parity` | Assert corr / mean / exact |

Canonical smoke case: `tests/fixtures/runtime_golden_a_red_chair_steps2.json`  
(`prompt="a red chair"`, `seed=7`, `steps=2`, `sigma_min=0.001`, `sigma_max=160`, `grid=16`).

---

## L0 — Upstream Torch ↔ MLX (existing)

From `shap_e/src` (activate `.mlx_venv` first):

```bash
cd shap_e/src
python -m cli.check.check_text300m_parity
python -m cli.check.check_mlx_diffusion_step_parity --prompt "a red chair"
python -m cli.check.check_mlx_karras_loop_parity --prompt "a red chair" --steps 2
python -m cli.check.check_mlx_decoder_params_parity
python -m cli.check.check_mlx_mesh_renderer_parity --latent-npz PATH.npz
```

These stay the gate when re-exporting weights. Swift does **not** compare to Torch directly.

Optional preflight in the one-shot script:

```bash
SHAP_E_PARITY_PREFLIGHT=1 ./shap_e/scripts/run_parity.sh
```

---

## L1 — MLX golden summary (existing)

```bash
cd shap_e/src
python -m cli.check.check_mlx_runtime_golden
# optional: --update to rewrite the committed fixture after intentional changes
```

Pytest (gated):

```bash
MLX_SHAP_E_RUN_GOLDENS=1 pytest tests/test_runtime_golden.py
```

MC unit (Torch ↔ MLX Python):

```bash
pytest tests/test_marching_cubes.py
```

---

## L2 — Swift ↔ MLX full tensors (this harness)

### One-shot

```bash
# repo root
source .mlx_venv/bin/activate
./shap_e/scripts/run_parity.sh
```

### Manual steps

```bash
# 1) Python MLX dump (writes shap_e/parity/python_*.npy + case.json)
cd shap_e/src
python -m cli.check.dump_mlx_swift_parity --check-golden

# 2) Swift dump (xcodebuild for Metal metallib)
cd ../swift
xcodebuild -scheme ShapECLI -destination 'platform=macOS' \
  -derivedDataPath ./.derivedData build
../swift/.derivedData/Build/Products/Debug/ShapECLI --parity

# 3) Compare (exit 0 = pass)
cd ../src
python -m cli.check.compare_mlx_swift_parity
```

### Tensor index

| File stem | Shape (smoke) | Gate |
|-----------|---------------|------|
| `token_ids` | `[1,77]` i32 | exact |
| `clip_embedding` | `[1,768]` f32 | corr ≥ 0.999, mean ≤ 1e-3 |
| `init_x` | `[1,1048576]` | shared input (Python→Swift); not compared |
| `sigmas` | `[steps+1]` | allclose atol 1e-5 |
| `latent` / `pred_xstart` | `[1,1048576]` | corr ≥ 0.999, mean ≤ 1e-3 |
| `params_*` | decoder param keys | corr / mean |
| `sdf_field` | `[G,G,G]` unpadded | corr / mean |
| `verts` / `faces` | variable | CPU: verts corr ≥ 0.999 / mean ≤ 1e-3 and exact face identity; Metal (`--metal-mc`): counts + bbox only |

Generated dumps are **gitignored**. Keep this README and regenerate with the scripts.

---

## Implementation notes

- Dump reuses `RuntimeGoldenCase` + `MLXShapERuntime` via `mlx_shap_e.runtime_core.parity_tensors`.
- SDF export: `MLXSTFMeshRenderer.extract_sdf_fields` / Swift `STFMeshRenderer.extractSDFFields`.
- Mesh MC path: CPU by default; `ShapECLI --parity --metal-mc` optional.
- Compare reuses colored bands conceptually from `cli/check/color.py` and is **assert-grade** (unlike L0).

### Parity alignment details (documented so we do not re-break them)

1. **Shared init noise** — Python uses `np.random.default_rng(seed).standard_normal * sigma_max`. Swift production uses `MLXRandom` (different stream). L2 dumps `python_init_x.npy`; `ShapECLI --parity` loads it into `KarrasSampler.sample(initX:)`.
2. **CFG uncond** — Python MLX Karras uses `zeros_like(embeddings)` for the uncond branch (not `embed("")`). Parity CLI matches zeros. Production `generate(prompt:)` and default `--ab-mesh` use `embed("")`. `--ab-mesh --uncond-zeros` is the Python-matched dump. Do not flip production to zeros to “fix holes” — that changes the whole SDF, not Metal vs CPU topology.
3. **Volume grid integer div** — Swift must use host integer `/` and `%` when building the G³ lattice. MLXArray `/` is float division and scrambles xyz (fixed in `volumeQueryPoints`).
4. **L1 family versions** — `compare_runtime_summary` requires expected family keys/versions and allows extra families (e.g. `image300m`) on the bundle.

### Smoke results (2026-07, `a red chair` steps=2 grid=16)

All L2 stages PASS after the fixes above (latent corr ≥ 0.9999, SDF corr ≥ 0.9999, exact mesh vert/face counts).

---

## Image conditioning — upstream Torch ↔ Swift

The Python MLX production runtime is text-only, so the image harness uses
upstream OpenAI Torch CLIP and `CLIPImageGridPointDiffusionTransformer` as the
independent oracle for Swift `ImageProcessor`, `ClipImageEmbedder`, and
`Image300MTransformer`.

One-shot from the repo root:

```bash
source .mlx_venv/bin/activate
./shap_e/scripts/run_image_parity.sh

# Real-image case (the default is deterministic synthetic RGB):
SHAP_E_IMAGE_PATH="$PWD/TripoSR/examples/chair.png" \
SHAP_E_IMAGE_PARITY_DIR="${TMPDIR:-/tmp}/shap-e-image-parity-chair" \
  ./shap_e/scripts/run_image_parity.sh

# Component tensors only:
SHAP_E_SKIP_MESH=1 ./shap_e/scripts/run_image_parity.sh
```

Stages and gates:

| Stage | Reference | Swift | Gate |
|---|---|---|---|
| Preprocess | OpenAI CLIP `preprocess` | `ImageProcessor.processCLIPImage` | corr ≥ 0.99, mean abs ≤ 0.075 (different platform resampling) |
| Vision grid | Torch `embed_images_grid` path | `ClipImageEmbedder.embedImagesGrid` with the exact shared tensor | corr ≥ 0.999, mean abs ≤ 1e-4 |
| Image300M, isolated | Torch inner transformer with Torch grid | Swift transformer with Torch grid | corr ≥ 0.999, mean abs ≤ 1e-4 |
| Image300M, full chain | Torch inner transformer | Swift transformer with Swift grid | corr ≥ 0.999, mean abs ≤ 1e-4 |
| Mesh smoke | Same Swift SDF | CPU and Metal marching cubes | Nonempty, identical counts, zero boundary/nonmanifold/invalid faces |

Manual entry points are
`python -m cli.check.dump_image_swift_parity`,
`ShapECLI --image-parity`, and
`python -m cli.check.compare_image_swift_parity`. The mesh smoke uses the
Swift-processed image, a shared initial latent, 64 sampling steps, guidance 3,
and grid 64 by default. Set `SHAP_E_IMAGE_BUNDLE`, `SHAP_E_IMAGE_CACHE`, or
`SHAP_E_IMAGE_PARITY_DIR` to override locations.

2026-11-17 results on the synthetic case and `TripoSR/examples/chair.png`
both passed. Image-grid mean absolute error was about `7.1e-6`; isolated and
full-chain Image300M mean absolute errors were `1.0e-6`–`2.4e-6`, all with
correlation printed as 1.00000. This run found and fixed a vertical flip in
`ImageProcessor`: CoreGraphics bitmap data was already in the raster order
expected by CLIP, so the extra context flip reversed rows. It is covered by
`ImageProcessorTests.testImageOrientationIsPreserved`.
