# File structure

This document is the canonical map of the code repository, the generated files
that sit beside it, the Shap-E runtime-bundle format, and the published
Hugging Face bundle layout. Architecture and data flow are drawn in the
[knowledge graph](../knowledge_graph/README.md); current release identifiers
and publication gates live in [Release status](RELEASE-STATUS.md).

## Four separate storage boundaries

| Boundary | What it owns | Versioned by |
|---|---|---|
| This GitHub repository | Source code, tests, docs, the tiny random-weight Swift fixture | Git commit on `main` |
| `TripoSR/` | TripoSR Python product and a second Swift package | The separate `signalnotnoise/TripoSR` Git repository, pinned here as a submodule |
| Local ignored directories | Downloaded Torch checkpoints, the full converted `runtime_bundle/`, virtual environments, Swift/Xcode build products, parity dumps, mesh outputs | Nothing; regenerate or download them |
| Hugging Face `signalNotNoise/shap-e-mlx` | Downloadable copy of the full Shap-E runtime bundle, BPE vocabulary, provenance, checksums, licenses, and validation summaries | Hugging Face Git revision |
| Hugging Face `signalNotNoise/triposr-mlx` (public) | Canonical `triposr_swift_ready.safetensors`, provenance, checksums, licenses, model card, and validation summaries | Hugging Face Git revision |

Do not confuse the three kinds of Shap-E bundle:

- `runtime_bundle/` is the full local format-v1 export consumed by the Python
  and Swift loaders.
- `shap_e/swift/Tests/ShapERuntimeTests/Fixtures/runtime_bundle_stub/` is a
  checked-in, random-valued, 28-tensor test fixture with no trained weights.
- `shap_e/Niko/artifacts/runtime_bundle_v1/`, if present locally, is an older
  ignored five-family export; it is not a source directory and is not tracked.

## Tracked GitHub repository

Counts below are tracked files at the documentation audit. Generated local
directories are listed separately so this tree also describes a fresh clone.

```text
apple-mlx-foundations/
├── AGENTS.md                         Canonical coding standards + graph rule
├── BUGS.md                           Open and historical runtime issues
├── CLAUDE.md                         Pointer to AGENTS.md
├── CONTRIBUTING.md                   Setup, hooks, and verification workflow
├── LICENSE                           MIT license for this repository
├── NOTICE                            Third-party attribution and weight notice
├── README.md                         Project overview and quick start
├── pyproject.toml                    Python package, Ruff, and dependency config
├── .swift-format                     Swift formatting/lint configuration
├── .gitmodules                       TripoSR submodule URL
├── .githooks/pre-commit              Standards check + knowledge-graph refresh
├── .github/
│   ├── copilot-instructions.md       Pointer to AGENTS.md
│   └── pull_request_template.md      Definition-of-done evidence checklist
├── scripts/
│   ├── check_standards.py            Deterministic coding-standards gate
│   ├── package_triposr_weights.py    Stages the TripoSR weights release bundle
│   └── verify.sh                     Unified local verification entry point
├── .cursor/rules/                    Cursor pointer to the graph rule
├── .vscode/launch.json               ShapECLI debug/release launch definitions
├── docs/
│   ├── CODING-STANDARDS.md           Detailed coding policy and enforcement
│   ├── FILE-STRUCTURE.md             This canonical layout and bundle contract
│   └── RELEASE-STATUS.md             Candidate revisions, evidence, and gates
├── knowledge_graph/
│   ├── README.md                     Curated architecture picture and query guide
│   ├── curated.json                  Hand-edited architectural meaning
│   ├── graph.json                    Generated graph; do not hand-edit
│   ├── update_graph.py               Rebuilds graph.json and README counts
│   └── install_hooks.sh              Sets core.hooksPath to .githooks
├── runtime_wrapper/                  Optional Swift model-selection facade
├── shap_e/                           Shap-E Python, Swift, CLI, scripts, docs
├── tests/                            Python tests and committed golden fixture
└── TripoSR/                          Git submodule; see its own tree below
```

### Shap-E tree

```text
shap_e/
├── README.md                         Python run convention and key commands
├── MESH_AB.md                        CPU versus Metal mesh-hole investigation
├── MLX_RENDERER_PORT_PLAN.md         Historical renderer-port plan
├── SWIFT_RUNTIME_MIGRATION_CHECKLIST.md
│                                     Historical Python-to-Swift port record
├── install.sh                        Legacy ad hoc Python package install helper
├── scripts/
│   ├── run_parity.sh                 One-shot Python MLX to Swift L2 pipeline
│   ├── run_image_parity.sh           Torch oracle to Swift image parity + mesh
│   └── run_mesh_ab.sh                Builds ShapECLI and runs CPU/Metal A/B
├── parity/
│   └── README.md                     Only tracked file; generated dumps ignored
├── src/
│   ├── load_pytorch_to_mlx.py        Standalone conversion utility
│   ├── mlx_weight_remapper.py        Standalone weight-remapping utility
│   ├── cli/
│   │   ├── check/                    L0/L1 checks, L2 dump and comparator
│   │   ├── export/                   Per-family command-line export wrappers
│   │   ├── inspect/                  Checkpoint inspection utilities
│   │   └── run/                      Prompt, trajectory, and mesh commands
│   └── mlx_shap_e/                   Installable Python package
│       ├── models/
│       │   ├── artifacts/            Text300M artifact data types
│       │   ├── conversions/          Torch-to-MLX weight conversions
│       │   ├── diffusion/            Diffusion step and Karras sampler
│       │   ├── embeddings/           CLIP tokenizer, embedders, artifacts
│       │   ├── rendering/            STF renderer and marching cubes
│       │   ├── decoder_params.py     Latent to NeRSTF parameter projection
│       │   ├── shape.py              Earlier structural sketch; not production
│       │   └── text300m.py           Diffusion transformer
│       ├── offline_tools/
│       │   └── runtime_artifact_exports.py
│       │                             Writes all seven bundle families
│       ├── runtime_artifacts/
│       │   ├── bundle.py             Python `RuntimeArtifactBundle`
│       │   └── manifest.py           Manifest dataclasses and tensor writer
│       ├── runtime_core/
│       │   ├── diffusion.py          Runtime diffusion configuration
│       │   ├── golden.py             L1 summary production/comparison
│       │   ├── parity_tensors.py     L2 tensor extraction
│       │   ├── pipeline.py           Python text prompt to latent/mesh runtime
│       │   └── types.py              Runtime result and version records
│       └── utils/cache_recovery.py   Corrupt-checkpoint retry helper
└── swift/                            SwiftPM package `ShapERuntime`
    ├── Package.swift
    ├── Package.resolved
    ├── README.md
    ├── run_tests.sh                  xcodebuild test entry point
    ├── Sources/
    │   ├── ShapECLI/main.swift       `--parity` and `--ab-mesh` executable
    │   └── ShapERuntime/
    │       ├── Models/embeddings/    CLIP tokenizer and text/vision embedders
    │       ├── *Artifact*.swift      Bundle-family data types
    │       ├── Runtime*.swift        Manifest, bundle loader, runtime types
    │       ├── TensorBlobLoader.swift Raw tensor validation/deserialization
    │       ├── *Transformer.swift    Text300M and Image300M model forwards
    │       ├── DiffusionStep.swift
    │       ├── KarrasSampler.swift
    │       ├── DecoderParamsProjector.swift
    │       ├── STFMeshRenderer.swift
    │       ├── MarchingCubes.swift / .metal
    │       ├── MetalMarchingCubes.swift
    │       ├── Mesh*.swift           Mesh result, topology, and OBJ export
    │       └── ShapERuntime.swift    Public text/image generation facade
    └── Tests/ShapERuntimeTests/
        ├── *.swift                   Unit and bundle-loading tests
        ├── RuntimeBundleFixture.swift Locates the checked-in stub or full bundle
        └── Fixtures/runtime_bundle_stub/
                                      28 random tensors plus manifest.json
```

There are 26 Python files under `mlx_shap_e`, 27 under `shap_e/src/cli`, 28
files under `shap_e/swift/Sources`, and 39 under
`shap_e/swift/Tests` (including fixture binaries). Counts drift as code
changes; paths and ownership are the durable contract.

### Other tracked packages

```text
runtime_wrapper/
├── Package.swift                     Depends on both local model packages
├── README.md
├── Sources/RuntimeWrapper/
│   ├── ThreeDGenerationRuntime.swift Backend selection facade
│   ├── ThreeDMeshResult.swift        Model-neutral result conversion
│   └── TripoSRMeshRuntime.swift      TripoSR fields through Shap-E meshing
└── Tests/RuntimeWrapperTests/

tests/
├── fixtures/runtime_golden_a_red_chair_steps2.json
├── test_download_shap_e_bundle.py
├── test_package_triposr_weights.py
├── test_swift_test_runner.py
├── test_clip_text_embedder.py
├── test_marching_cubes.py
├── test_mesh_batch_decode.py
└── test_runtime_golden.py
```

Swift package entry points are:

| Package | Manifest | Products | Sources |
|---|---|---|---|
| `ShapERuntime` | `shap_e/swift/Package.swift` | `ShapERuntime`, `ShapECLI` | `shap_e/swift/Sources` |
| `TripoSR` (consumer) | `TripoSR/Package.swift` | `TripoSRRuntime`, `TripoSRCLI` | `TripoSR/swift/Sources` |
| `TripoSR` (standalone) | `TripoSR/swift/Package.swift` | Same products | Same sources |
| `RuntimeWrapper` | `runtime_wrapper/Package.swift` | `RuntimeWrapper` | `runtime_wrapper/Sources` |

All four package definitions target macOS 14, iOS 17, and visionOS 1 and
declare `mlx-swift` from 0.29.1. The tracked `Package.resolved` files for
`ShapERuntime` and standalone TripoSR currently select 0.31.3. The root
`.gitignore` excludes `runtime_wrapper/Package.resolved`, so a clean clone
resolves that package again within its declared range.

### TripoSR submodule tree

`TripoSR` is one Gitlink in this repository, currently documented at commit
`6da4a3cd3f8454f7c98607249476a26c836509ee` in
[Release status](RELEASE-STATUS.md). Its own tracked layout is:

```text
TripoSR/
├── Package.swift                     Consumer manifest for RuntimeWrapper
├── run.py / gradio_app.py            Upstream Python product entry points
├── tsr/                              Upstream TSR system and model modules
├── examples/                         Input examples
├── figures/                          Upstream documentation images
├── scripts/                          Python parity dump/compare/run scripts
├── parity/README.md                  Generated parity dumps are ignored
├── export_tripo_weights.py           Upstream checkpoint to MLX pack
├── convert_weights_to_mlx.py         Key/layout remap for Swift
├── triposr_keys.txt                  Checkpoint key inventory
└── swift/
    ├── Package.swift                 Standalone manifest over the same sources
    ├── Sources/TripoSRRuntime/       DINOv2, LRM, triplane, NeRF query
    ├── Sources/TripoSRCLI/           Inference/parity executable
    └── Tests/TripoSRRuntimeTests/
```

TripoSR's detailed documentation stays in
[`TripoSR/README.md`](../TripoSR/README.md),
[`TripoSR/SCRIPTS.md`](../TripoSR/SCRIPTS.md), and
[`TripoSR/swift/README.md`](../TripoSR/swift/README.md). Changes inside the
submodule must be committed in that repository before this repository can pin
the resulting new Gitlink.

## Local, ignored, and generated files

A working development checkout commonly contains directories that are absent
from a fresh clone and must never be treated as source:

| Path | Contents | Recreate from |
|---|---|---|
| `.mlx_venv/` | Python environment auto-activated by parity scripts | `python3 -m venv .mlx_venv`, then install the project |
| `shap_e_model_cache/` | Upstream CLIP/Shap-E `.pt` checkpoints and YAML configs | OpenAI `clip` / `shap_e.models.download` loaders |
| `shap_e/shap_e_model_cache/` | Alternate cache location recognized by some tools | Same loaders, depending on working directory |
| `runtime_bundle/` | Full converted format-v1 bundle | `mlx_shap_e.offline_tools.runtime_artifact_exports:prepare_runtime_artifacts` or a downloaded Hugging Face snapshot |
| `TripoSR/triposr_weights/` | `triposr_mlx.safetensors` (intermediate pack, never published) and `triposr_swift_ready.safetensors` (canonical Swift artifact, published as `signalNotNoise/triposr-mlx`) | `TripoSR/export_tripo_weights.py`, then `TripoSR/convert_weights_to_mlx.py`; stage the release bundle with `scripts/package_triposr_weights.py` |
| `shap_e/parity/*.npy`, `*.npz`, summaries, `case.json` | Generated L2 dumps | `shap_e/scripts/run_parity.sh` |
| `TripoSR/parity/*.npy`, `*.npz` | Generated TripoSR parity dumps | `TripoSR/scripts/run_parity.sh` |
| `shap_e/mesh_ab/`, `shap_e/mesh_ab_smoke/` | Mesh A/B outputs | `shap_e/scripts/run_mesh_ab.sh` |
| `shap_e/outputs/`, `*.log`, `*.npz`, `*.safetensors`, `*.profraw` | Runs, logs, tensors, profiles, and scratch output | Individual commands; not release inputs unless a release record names them |
| `.build/`, `*/.build/`, `.derivedData/`, `.swiftpm/`, `xcuserdata/` | SwiftPM/Xcode products and user state | Xcode or SwiftPM |
| `__pycache__/`, `*.egg-info/`, tool caches | Python bytecode and packaging/test caches | Python tooling |

The exact ignore contract is [`.gitignore`](../.gitignore), plus ignore files
inside the submodule. A path being ignored does not prove it is disposable:
checkpoint and bundle directories can take a long time to recreate.

## Runtime-bundle format v1

Both production loaders resolve one directory through its root
`manifest.json`:

- Python: `mlx_shap_e.runtime_artifacts.manifest:load_manifest` and
  `mlx_shap_e.runtime_artifacts.bundle:RuntimeArtifactBundle`
- Swift: `RuntimeArtifactBundle(bundleURL:)`,
  `RuntimeArtifactBundle.manifestFileName`, `RuntimeArtifactManifest`, and
  `TensorBlobLoader.loadTensor(from:entry:family:)`

The on-disk JSON contract is:

```json
{
  "artifact_version": 1,
  "families": {
    "clip_text": {
      "version": 1,
      "metadata": {
        "context_length": 77
      },
      "tensors": {
        "token_embedding.weight": {
          "name": "token_embedding.weight",
          "path": "clip_text/token_embedding__weight.bin",
          "shape": [49408, 768],
          "dtype": "float32"
        }
      }
    }
  }
}
```

Format rules implemented by
`mlx_shap_e.runtime_artifacts.manifest:save_tensor_family` and validated by
Swift `TensorEntry`, `ArtifactFamilyManifest`, and `RuntimeTensor` are:

1. `artifact_version` and every family `version` are currently `1`.
2. Tensor paths are bundle-relative and normally `<family>/<safe-name>.bin`.
   The writer replaces both `/` and `.` in a tensor name with `__`.
3. A `.bin` file is the tensor's raw, contiguous C-order bytes with no header,
   produced by NumPy `array.tobytes(order="C")`. Despite the extension, it is
   not a PyTorch checkpoint.
4. `shape` and `dtype` in `manifest.json` are authoritative. Swift accepts
   `float32`, `int32`, and `int64`; all 1,088 tensors in the current Shap-E
   release bundle are `float32`.
5. Swift rejects negative dimensions, shape-product overflow, unsupported
   dtypes, missing files, and exact byte-count mismatches. Python reshapes
   each file with `np.fromfile(..., dtype=...)` and the manifest shape.
6. The loaders do not verify hashes. Release integrity comes from the Hub
   revision and the sibling `checksums.sha256`, not from `manifest.json`.

### Families and loaders

The sizes and counts below describe the current full release bundle, not the
tiny test stub. Byte totals are the sums of tensor files in each directory.

| Family | Role / required metadata | Python loader | Swift loader | Tensors | Bytes |
|---|---|---|---|---:|---:|
| `clip_text` | ViT-L/14 text tower; `clip_name`, `context_length`, `tokenizer`, `transformer_width`, `transformer_heads`, `transformer_layers` | `load_clip_text_artifact()` | `loadClipTextArtifact()` | 149 | 494,601,216 |
| `clip_vision` | ViT-L/14 vision tower; `clip_name`, `input_resolution`, `patch_size`, `transformer_width`, `transformer_heads`, `transformer_layers`, `output_dim` | No typed loader; generic `load_family_tensors()` only | `loadClipVisionArtifact()` | 296 | 1,215,864,832 |
| `text300m` | Text-conditioned diffusion transformer; `n_ctx`, `d_latent` | `load_text300m_artifact()` | `loadText300MArtifact()` | 303 | 1,262,768,128 |
| `image300m` | Image-conditioned diffusion transformer; `n_ctx`, `d_latent` | No typed loader; generic `load_family_tensors()` only | `loadImage300MArtifact()` | 305 | 1,263,824,896 |
| `diffusion` | Noise schedules and posterior coefficients; `num_timesteps`, `model_mean_type` | `load_diffusion_config()` | `loadDiffusionConfig()` | 7 | 28,672 |
| `decoder_params` | Latent projections to NeRSTF parameters; `param_shapes`, `latent_ctx`, `latent_warp_name`, `latent_warp_coeff1`, `ln_eps` | `load_decoder_params()` | `loadDecoderParamsArtifact()` | 16 | 903,712,488 |
| `renderer` | Static NeRSTF weights, bounding box, field settings; `grid_size` and activation/channel metadata | `load_renderer()` | `loadRendererArtifact()` | 12 | 595,016 |

Total manifest tensors: **1,088**. The text path requires
`clip_text`, `text300m`, `decoder_params`, `diffusion`, and `renderer`
(2,661,705,520 tensor bytes). Swift image generation additionally requires
`clip_vision` and `image300m` (2,479,689,728 tensor bytes). Loaders request
families independently; the pinned downloader's `--text-only` snapshot has
also been tested through a text mesh. The authoritative fresh-download image
and text validation records are in [Release status](RELEASE-STATUS.md).

The generic entry points are Python
`RuntimeArtifactBundle.load_family_tensors(family)` and Swift
`RuntimeArtifactBundle.loadFamilyTensors(_:)`. Typed loading validates the
required metadata and named tensors needed by each runtime. The BPE
vocabulary is deliberately not a tensor family or manifest entry: local
exports leave `bpe_simple_vocab_16e6.txt.gz` outside `runtime_bundle/`, while
the Hugging Face distribution places that separate file beside
`manifest.json`. Swift still receives bundle and BPE URLs independently via
`ShapERuntime(modelBundleURL:bpeVocabularyURL:)`.

The exporter source of truth is
`shap_e/src/mlx_shap_e/offline_tools/runtime_artifact_exports.py`:
`export_clip_text_runtime_family`, `export_clip_vision_runtime_family`,
`export_text300m_runtime_family`, `export_image300m_runtime_family`,
`export_diffusion_runtime_family`, and `export_decoder_runtime_families`
(the last writes both `decoder_params` and `renderer`). Its
`prepare_runtime_artifacts(bundle_dir:cache_dir:clip_name:)` function writes
all seven families.

### Checked-in test stub

`runtime_bundle_stub/` mirrors only five families—`clip_text`,
`decoder_params`, `diffusion`, `renderer`, and `text300m`—with 28 tensors in
29 files including `manifest.json`. It exists to test bundle parsing and
construction paths, not numerical model output. There is currently no
checked-in `--stub` generator command; if the schema changes, update the stub
deliberately and keep its [fixture README](../shap_e/swift/Tests/ShapERuntimeTests/Fixtures/README.md)
in sync.

## Hugging Face model repository

The downloadable distribution is
[`signalNotNoise/shap-e-mlx`](https://huggingface.co/signalNotNoise/shap-e-mlx).
At the audited revision recorded in [Release status](RELEASE-STATUS.md), it
contains **1,104 files totaling 5,143,254,750 bytes (5.143 GB / 4.790 GiB)**:

```text
signalNotNoise/shap-e-mlx/
├── .gitattributes                    Hub large-file/Xet tracking
├── LICENSE                           MIT license for the converted bundle
├── README.md                         Model card
├── manifest.json                     335,587 bytes; format-v1 contract
├── bpe_simple_vocab_16e6.txt.gz      1,356,917 bytes; separate CLIP vocab
├── provenance.json                   Source checkpoint hashes and code pin
├── checksums.sha256                  1,101 delivery-file entries
├── reproduce_export.py               Re-export driver
├── requirements-validation.txt       Pinned validation environment
├── clip_text/          149 .bin      494,601,216 bytes
├── clip_vision/        296 .bin    1,215,864,832 bytes
├── text300m/           303 .bin    1,262,768,128 bytes
├── image300m/          305 .bin    1,263,824,896 bytes
├── decoder_params/      16 .bin      903,712,488 bytes
├── diffusion/            7 .bin           28,672 bytes
├── renderer/            12 .bin          595,016 bytes
├── licenses/
│   ├── CLIP.txt
│   └── Shap-E.txt
└── validation/
    ├── download-verification.json
    ├── l0-results.json
    ├── l2.log
    └── mesh-report.json
```

The 1,102 entries in `checksums.sha256` cover every Hub file except
`checksums.sha256` itself and the Hub-generated `.gitattributes`; together
those two account for the 1,103-file total. A local upload staging
directory may also contain `.cache/huggingface/upload/` bookkeeping files;
those are not repository files and are not part of the Hub tree.

No source code needed to use the bundle lives only on Hugging Face. The Hub
`reproduce_export.py` is a release driver; model and loader source is in this
GitHub repository, and `provenance.json` pins the tested code commit.

### TripoSR weights repository (public)

`signalNotNoise/triposr-mlx` follows the same pattern at a smaller scale.
At the pinned revision in [Release status](RELEASE-STATUS.md) it contains
12 files: `.gitattributes` plus the 11 staged files —
`triposr_swift_ready.safetensors` (499 tensors, 1,674,793,528 bytes),
`provenance.json`, `checksums.sha256` (10 entries: every staged file except
itself), `LICENSE`, `NOTICE`, `README.md` (model card),
`reproduce_export.py`, `licenses/TripoSR.txt`, and
`validation/{parity-summary,mesh-smoke,download-verification}.json`.
The intermediate `triposr_mlx.safetensors` is not in the repository.
Staging is produced by `scripts/package_triposr_weights.py`, which refuses
to reuse an existing staging directory.

## Documentation map

| Need | Canonical document |
|---|---|
| Project overview, checkout, requirements, quick start | [`README.md`](../README.md) |
| Coding policy, workflow, and definition of done | [`CODING-STANDARDS.md`](CODING-STANDARDS.md), summarized in [`AGENTS.md`](../AGENTS.md) |
| Contributor setup and verification commands | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| File ownership, generated paths, and bundle format | This document |
| Release revisions, proof, known gaps, and publication gates | [Release status](RELEASE-STATUS.md) |
| Architecture, module relationships, and port map | [`knowledge_graph/README.md`](../knowledge_graph/README.md) |
| Swift package API and tests | [`shap_e/swift/README.md`](../shap_e/swift/README.md) |
| Python run convention and key commands | [`shap_e/README.md`](../shap_e/README.md) |
| L0/L1/L2 parity harness and tensor gates | [`shap_e/parity/README.md`](../shap_e/parity/README.md) |
| CPU/Metal mesh-hole diagnosis | [`shap_e/MESH_AB.md`](../shap_e/MESH_AB.md) |
| Shap-E/TripoSR selection facade | [`runtime_wrapper/README.md`](../runtime_wrapper/README.md) |
| TripoSR product and scripts | [`TripoSR/README.md`](../TripoSR/README.md), [`TripoSR/SCRIPTS.md`](../TripoSR/SCRIPTS.md) |
| TripoSR Swift package | [`TripoSR/swift/README.md`](../TripoSR/swift/README.md) |
| Third-party notices | [`NOTICE`](../NOTICE) |
| Open issue record | [`BUGS.md`](../BUGS.md) |
| Historical renderer plan | [`shap_e/MLX_RENDERER_PORT_PLAN.md`](../shap_e/MLX_RENDERER_PORT_PLAN.md) |
| Historical Swift migration record | [`shap_e/SWIFT_RUNTIME_MIGRATION_CHECKLIST.md`](../shap_e/SWIFT_RUNTIME_MIGRATION_CHECKLIST.md) |

The historical plan/checklist are retained as project history. Current code,
package manifests, this map, and the knowledge graph override any path or
status statements in them.

## Keeping this map current

Update this file when a directory, package, artifact family, support matrix,
or external distribution changes. Recompute tracked paths with
`git ls-files`, bundle tensors from `manifest.json`, and remote files/bytes
from the exact Hugging Face revision being described. Run
`python3 knowledge_graph/update_graph.py` after repository changes, as
required by [`AGENTS.md`](../AGENTS.md).
