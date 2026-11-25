# Apple MLX Foundations: Shap-E VisionOS Port

A comprehensive MLX-native workspace dedicated to porting OpenAI's [Shap-E](https://github.com/openai/shap-e) text-to-3D diffusion model into a hardware-accelerated Apple Silicon pipeline.

The ultimate goal of this project is to lay the complete groundwork for deploying state-of-the-art interactive 3D generation locally on Apple Vision Pro via Swift and `visionOS`.

## 🚀 Key Achievements

### 1. MLX-Native Swift Framework (`ShapERuntime`)
We have successfully completed the migration of the entire Shap-E pipeline into a native Swift framework. This includes the CLIP embedder, the 300M parameter diffusion transformer, and the STF renderer—all running natively on Metal via MLX.

### 2. GPU-Optimized Marching Cubes
The legacy `skimage` and NumPy dependencies have been eliminated. We implemented a high-performance, deterministic Marching Cubes SDF extractor in Swift that handles millions of vertices (e.g., 1.3M+ for complex scenes) using chunked evaluation to ensure GPU stability.

### 3. Flawless Mathematical Parity
Our runtime is mathematically locked to the original OpenAI PyTorch implementation. We identified and resolved critical precision drifts in:
*   **GELU Activations**: Replaced approximate QuickGELU with standard mathematical GELU for backbone fidelity.
*   **Karras Heun Sampler**: Re-aligned noise schedules to match original PyTorch integrators.
*   **Latent Unwarping**: Corrected `tan2` polynomial scaling for accurate geometry projection.

### 4. Production-Ready CLI & Color Export
We built a Swift CLI (`ShapECLI`) for MLX ↔ Swift parity dumps and CPU-vs-Metal mesh A/B runs, with seed control and high-resolution mesh extraction with full vertex color support (`v x y z r g b`).

### 5. Layered Python ↔ Swift parity (Shap-E)

| Layer | What | Command |
|-------|------|---------|
| L0 | Torch ↔ Python MLX | `python -m cli.check.check_mlx_*` (existing) |
| L1 | MLX golden JSON | `python -m cli.check.check_mlx_runtime_golden` |
| L2 | Full tensors MLX ↔ Swift | `./shap_e/scripts/run_parity.sh` |

Details: [`shap_e/parity/README.md`](shap_e/parity/README.md). L2 shares the golden case, dumps `python_*.npy` / `swift_*.npy`, and is assert-grade.

---

## Checkout

`TripoSR` is a git submodule. `RuntimeWrapper` depends on `TripoSR/Package.swift`, so a clone that does not check it out cannot resolve that package.

```bash
git clone --recurse-submodules https://github.com/signalnotnoise/apple-mlx-foundations.git
# existing clone (sync first so a cached pre-HTTPS submodule URL is refreshed):
git submodule sync
git submodule update --init
```

## Requirements

- Apple Silicon Mac with a recent Xcode (the Swift packages are built with xcodebuild so MLX's `default.metallib` is bundled).
- Python ≥ 3.10 for the MLX reference implementation and export/check tools. The parity scripts auto-activate `.mlx_venv` at the repo root when it exists:

```bash
python3 -m venv .mlx_venv
source .mlx_venv/bin/activate
pip install -e .
```

## Model weights

No pretrained weights are included in this Git repository (see
[`NOTICE`](NOTICE)). Both runtimes need the same format-v1 bundle, whose full
layout, manifest schema, tensor naming, family counts, and byte sizes are in
[`docs/FILE-STRUCTURE.md`](docs/FILE-STRUCTURE.md#runtime-bundle-format-v1).

A converted v0.1.0 bundle is published as the public Hugging Face model
[`signalNotNoise/shap-e-mlx`](https://huggingface.co/signalNotNoise/shap-e-mlx).
The pinned revision, validation evidence, and remaining release gates are
recorded in [`docs/RELEASE-STATUS.md`](docs/RELEASE-STATUS.md). Text and Swift
image component/mesh validation pass.

Download the pinned public bundle into an empty bundle directory; the
downloader verifies every downloaded checksum entry:

```bash
cd shap_e/src
python -m cli.run.download_shap_e_bundle
# Text families only (omits clip_vision + image300m):
python -m cli.run.download_shap_e_bundle --text-only --output /path/to/text-bundle
```

The downloader is pinned in
`cli.run.download_shap_e_bundle.DEFAULT_REVISION`. Text-only mode refuses an
output directory that already contains image-family tensors, so it cannot
silently turn an existing full bundle into a mislabeled partial one.

To build the bundle yourself:

- Download the original Shap-E and CLIP checkpoints through the upstream
  [`shap-e`](https://github.com/openai/shap-e) and CLIP loaders into
  `shap_e_model_cache/` (gitignored).
- Run the all-family exporter rather than only the per-family wrappers:

  ```bash
  cd shap_e/src
  python -m mlx_shap_e.offline_tools.runtime_artifact_exports \
    --bundle-dir ../../runtime_bundle \
    --cache-dir ../../shap_e_model_cache
  ```

  `prepare_runtime_artifacts` writes `clip_text`, `clip_vision`, `text300m`,
  `image300m`, `diffusion`, `decoder_params`, and `renderer` under
  `runtime_bundle/` (gitignored; 5.143 GB for the full audited bundle).
- Obtain `bpe_simple_vocab_16e6.txt.gz` from OpenAI's
  [CLIP](https://github.com/openai/CLIP) project. A local export leaves it
  separate from `runtime_bundle/`; the Hugging Face distribution places the
  same separate file beside `manifest.json`. Swift still takes the bundle and
  BPE URLs independently.

## 🛠 Project Layout

```text
shap_e/src/
  mlx_shap_e/                    # Python MLX Reference Implementation
  cli/                           # check / export / inspect / run entry points
shap_e/swift/                    # Shap-E visionOS/Swift Framework (SwiftPM package)
  Sources/ShapERuntime/          # Embedder, Transformer, Sampler, Renderer, Metal MC
  Sources/ShapECLI/              # Parity / mesh A/B command-line tool
  Tests/ShapERuntimeTests/
    Fixtures/
      runtime_bundle_stub/       # Tiny random-weight stub for unit tests (checked in)
runtime_bundle/                  # Pre-converted full model weights — gitignored, multi-GB

TripoSR/                         # TripoSR (image → 3D) — Python upstream + Swift/MLX port
  tsr/                           # Official Python model
  swift/                         # MLX Swift runtime (DINO + LRM + triplane + NeRF query)
  scripts/                       # dump_python_parity.py, compare_parity.py, run_parity.sh
  parity/                        # Generated golden / Swift tensors (regenerate; see parity/README.md)
  convert_weights_to_mlx.py      # Safetensors remapping for Swift

runtime_wrapper/                 # Optional facade depending on both independent runtimes
tests/                           # Python tests + committed L1 golden fixture
docs/                            # Coding standards, file map, and release status
scripts/                         # Standards checker and unified verification
knowledge_graph/                 # Generated architectural map + curated source
```

This is an abbreviated tree. The complete tracked layout, ignored/generated
directories, Swift package matrix, TripoSR submodule layout, runtime-bundle
schema, and Hugging Face tree are maintained in
[`docs/FILE-STRUCTURE.md`](docs/FILE-STRUCTURE.md).

## Documentation

| Topic | Document |
|---|---|
| Coding standards and definition of done | [`docs/CODING-STANDARDS.md`](docs/CODING-STANDARDS.md) (summary in [`AGENTS.md`](AGENTS.md)) |
| Contributor setup and verification | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Repository, local-artifact, bundle, and Hugging Face layouts | [`docs/FILE-STRUCTURE.md`](docs/FILE-STRUCTURE.md) |
| Release candidate revisions, validation scope, and public-release gates | [`docs/RELEASE-STATUS.md`](docs/RELEASE-STATUS.md) |
| Architecture and Python-to-Swift port map | [`knowledge_graph/README.md`](knowledge_graph/README.md) |
| Swift package API and tests | [`shap_e/swift/README.md`](shap_e/swift/README.md) |
| Python commands | [`shap_e/README.md`](shap_e/README.md) |
| L0/L1/L2 parity gates | [`shap_e/parity/README.md`](shap_e/parity/README.md) |
| CPU/Metal mesh diagnosis | [`shap_e/MESH_AB.md`](shap_e/MESH_AB.md) |
| Model-selection facade | [`runtime_wrapper/README.md`](runtime_wrapper/README.md) |
| Third-party attribution | [`NOTICE`](NOTICE) |

### TripoSR scripts (summary)

**Entry-point map (product vs weights vs parity):** [`TripoSR/SCRIPTS.md`](TripoSR/SCRIPTS.md)

| Goal | Command |
|------|---------|
| Mesh (Python product) | `python TripoSR/run.py examples/chair.png --output-dir output/` |
| Mesh (Swift) | `RuntimeWrapper` (`TripoSRMeshRuntime`) — see below |
| Gradio demo | `python TripoSR/gradio_app.py` |
| Swift weights | `python TripoSR/export_tripo_weights.py` then `convert_weights_to_mlx.py`, or the public `signalNotNoise/triposr-mlx` bundle (see [`docs/RELEASE-STATUS.md`](docs/RELEASE-STATUS.md)) |
| Parity PyTorch ↔ Swift | `./TripoSR/scripts/run_parity.sh` |

Swift details: [`TripoSR/swift/README.md`](TripoSR/swift/README.md).
- Build with **xcodebuild** (plain `swift run` often misses MLX `default.metallib`)
- Parity is high-correlation, not bit-exact.

### Select Shap-E or TripoSR from Swift

The separate `RuntimeWrapper` package depends on both model runtimes and returns
`ThreeDMeshResult` for either backend. `ShapERuntime` and `TripoSRRuntime` remain independent.
Shap-E supports text and images; TripoSR is the image-only alternative.

```swift
import RuntimeWrapper

let runtime = try ThreeDGenerationRuntime(
    configuration: .tripoSR(weightsURL: tripoSRWeightsURL)
)
let meshes = try await runtime.generate(image: cgImage)

// Or retain Shap-E text/image generation:
let shapE = try ThreeDGenerationRuntime(
    configuration: .shapE(
        modelBundleURL: modelBundleURL,
        bpeVocabularyURL: bpeVocabularyURL
    )
)
```

TripoSR expects an isolated subject. Transparent input is composited over the model's
gray background, but automatic background removal is not part of the Swift runtime.

### Shap-E mesh hole A/B (CPU vs Metal)

When Vision meshes look holey, isolate model vs Metal MC:

```bash
./shap_e/scripts/run_mesh_ab.sh --prompt "a ford mustang" --seed 859729 --steps 64 --grid 128
```

Docs: [`shap_e/MESH_AB.md`](shap_e/MESH_AB.md).

## ⚙️ Quick Start (Swift CLI)

`ShapECLI` has two modes, `--parity` and `--ab-mesh`. Build it with xcodebuild (plain
`swift run` often misses MLX's `default.metallib`), or use the wrapper scripts, which
build and run it for you:

```bash
# From the repo root. Requires runtime_bundle/ and the CLIP BPE vocab.
./shap_e/scripts/run_mesh_ab.sh --prompt "a fancy castle" --steps 64 --grid 128
./shap_e/scripts/run_parity.sh

# Swift unit tests:
cd shap_e/swift && ./run_tests.sh
```

---

## 📦 Release status

There is no tagged GitHub release yet. The current Shap-E v0.1.0 candidate and
its private Hugging Face weights, evidence, and remaining gates are tracked in
[`docs/RELEASE-STATUS.md`](docs/RELEASE-STATUS.md).

Implemented in the current codebase:

*   **Fixed**: `broadcast_shapes` error during high-res Marching Cubes extraction.
*   **Fixed**: Structural geometry corruption caused by incorrect GELU activation function.
*   **Added**: Full vertex color (RGB) export support in OBJ files.
*   **Added**: 4096-point chunking for vertex evaluators to prevent GPU memory faults.
*   **Added**: Deterministic seeding for reproducible 3D generation.

## 🐞 Active Development
Remaining architectural refinements are documented in [`BUGS.md`](./BUGS.md).

## License

MIT — see [`LICENSE`](LICENSE) (© signalnotnoise). This is an independent port of
OpenAI's Shap-E; it is not affiliated with or endorsed by OpenAI or Apple.
Third-party attribution for Shap-E, CLIP, TripoSR, and MLX is in
[`NOTICE`](NOTICE). The `TripoSR` submodule carries its own MIT license.
