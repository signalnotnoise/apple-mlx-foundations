# ShapERuntime

A native Swift + [MLX](https://github.com/ml-explore/mlx-swift) port of OpenAI's
[Shap-E](https://github.com/openai/shap-e) text-/image-to-3D pipeline for Apple
Silicon: CLIP tokenizer and embedders, the 300M diffusion transformers, a Karras
Heun sampler, decoder-parameter projection, and an STF mesh renderer with CPU
and Metal marching cubes. Part of
[apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations);
see the root `NOTICE` for third-party attribution. The complete source and
weight-bundle layout is documented in
[`docs/FILE-STRUCTURE.md`](../../docs/FILE-STRUCTURE.md), and current release
revisions and validation limits are in
[`docs/RELEASE-STATUS.md`](../../docs/RELEASE-STATUS.md).

## Products

| Product | What |
|---------|------|
| `ShapERuntime` | Library: `ShapERuntime` end-to-end text/image → mesh API |
| `ShapECLI` | Executable for MLX ↔ Swift parity dumps (`--parity`) and CPU-vs-Metal mesh A/B (`--ab-mesh`) |

Platforms: macOS 14+, iOS 17+, visionOS 1+. Built with Xcode (plain
`swift build`/`swift run` can miss MLX's `default.metallib`).

## What you need at runtime

Model weights are **not** included in this package or repository. The runtime
loads:

- a converted format-v1 `runtime_bundle/` directory (`manifest.json` plus the
  `clip_text`, `clip_vision`, `text300m`, `image300m`, `diffusion`,
  `decoder_params`, and `renderer` tensor families), produced from the
  original Shap-E checkpoints by
  `mlx_shap_e.offline_tools.runtime_artifact_exports:prepare_runtime_artifacts`,
  or downloaded as a complete Hugging Face snapshot, and
- the CLIP BPE vocabulary `bpe_simple_vocab_16e6.txt.gz` from OpenAI's
  [CLIP](https://github.com/openai/CLIP) project. It is a separate root file,
  not a manifest family; the Hugging Face bundle includes it beside
  `manifest.json`.

The manifest keys, raw tensor byte layout, loader functions, family counts,
and exact Hugging Face tree are specified in
[`docs/FILE-STRUCTURE.md`](../../docs/FILE-STRUCTURE.md#runtime-bundle-format-v1).
Fetch the pinned public candidate with
`python -m cli.run.download_shap_e_bundle` from `shap_e/src`; add
`--text-only` with an empty output directory for the five text-path families.
The downloaded directory is passed unchanged as `modelBundleURL`, with its
root BPE file as `bpeVocabularyURL`.
Text parity and the Swift image
Torch-parity/mesh harness pass, with metrics published in the Hub
`validation/image-parity.json`; remaining gates are in
[`docs/RELEASE-STATUS.md`](../../docs/RELEASE-STATUS.md).

Text weights load at init; the image pipeline (~2.3 GB) loads lazily on the
first `generate(image:)` call.

## Usage

```swift
import ShapERuntime

let runtime = try ShapERuntime(
    modelBundleURL: bundleURL,          // runtime_bundle/
    bpeVocabularyURL: bpeURL            // bpe_simple_vocab_16e6.txt.gz
)

// One mesh (seed defaults to 0):
let mesh = try await runtime.generate(prompt: "a red chair")

// A batch:
let meshes = try await runtime.generate(prompt: "a red chair", batchSize: 4)

// From an image:
let fromImage = try await runtime.generate(image: cgImage)
```

Results are `ShapEMeshResult` values wrapping a `MeshResult`
(vertices, faces, and vertex colors).

## Tests

```bash
./run_tests.sh                    # all tests (xcodebuild, ShapERuntime-Package scheme)
./run_tests.sh ImageProcessorTests
```

Unit tests run against the checked-in random-weight stub bundle in
`Tests/ShapERuntimeTests/Fixtures/`; golden/parity tests skip when the full
weights or CLIP vocabulary are not present locally.
