# Release status

Last audited: **2026-11-23**.

This is the canonical release record for the Shap-E v0.1.0 public release
and the TripoSR weights publication.
Repository layout
and the bundle file contract are documented in
[File structure](FILE-STRUCTURE.md).

## Current state

The Shap-E source and weights are public as v0.1.0. This source repository
starts with one clean initial commit seeded from the cleaned development
tree. The full development history is retained separately in a private
archive and is not part of this repository:

| Artifact | Location | Current state | Pinned revision |
|---|---|---|---|
| Source code | `signalnotnoise/apple-mlx-foundations` on GitHub | Public; matching `v0.1.0` tag and GitHub Release published | Initial commit `018215715177943fe72cea8e540bc7b80dcb593f`; anonymous recursive clean clone, full local verification, public bundle download, and mesh smoke passed before tagging |
| Shap-E weights | `signalNotNoise/shap-e-mlx` on Hugging Face | Public; matching `v0.1.0` tag | Pinned revision `d71f70a09b9c03535188d006a719edda5c19a6eb`; anonymous exact download verified against `checksums.sha256` |
| TripoSR source submodule | `signalnotnoise/TripoSR` on GitHub | Public repository; pinned as a submodule here | `6da4a3cd3f8454f7c98607249476a26c836509ee` |
| TripoSR MLX weights | `signalNotNoise/triposr-mlx` on Hugging Face | **Public** | Pinned revision `17a0c5286e2b2bb6fc9e3654ca31cfc9dc4b2041` (weights commit `8353300f4a032aca9099c09344292a8809c2eb58`, unchanged since the private upload); anonymous fresh download + checksums + mesh recheck passed |

## Shap-E v0.1.0 weights

The Hugging Face repository contains 1,104 files and 5,143,254,750 bytes,
including 1,088 float32 tensors in seven format-v1 families. Its card and
complete layout are summarized in
[File structure — Hugging Face model repository](FILE-STRUCTURE.md#hugging-face-model-repository).

Provenance committed beside the weights in `provenance.json` records:

- Code commit: `8ab95d13140fa617a50f4f7ccf7a56fec9768ef2`
- Bundle format: `artifact_version: 1`
- Upstream Shap-E code: `50131012ee11c9d2617f3886c10f000d3c7a3b43`
- Upstream CLIP code: `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`
- Source checkpoints, byte counts, and SHA-256 hashes for `ViT-L-14.pt`,
  `text_cond.pt`, `image_cond.pt`, `vector_decoder.pt`, and their YAML files
- Export module:
  `mlx_shap_e.offline_tools.runtime_artifact_exports`
- Environment: Python 3.14.3, macOS 27.3/arm64, Xcode 27.0 build 27A5252f,
  mlx-swift 0.31.3

Conversion is a format/layout export. It is not training or fine-tuning.
Upstream licenses are included on the Hub under `licenses/Shap-E.txt` and
`licenses/CLIP.txt`; project attribution is in [`NOTICE`](../NOTICE).

## Verification already recorded

The Hugging Face `validation/` directory carries the public-evidence subset:
`l0-results.json`, `l2.log`, `mesh-report.json`, and
`download-verification.json`. The fuller local release workpapers are not part
of this Git repository.

### Component and parity evidence

- **L0 component comparisons:** Text300M, CLIP text, and four decoder
  projection weights were compared against the upstream path. Recorded
  correlations are effectively 1.0; the largest recorded max-absolute error is
  Text300M at `1.5735626220703125e-05`.
- **L1:** The committed Python MLX golden summary passed.
- **L2:** Full-tensor Python MLX versus Swift passed for token IDs, CLIP
  embedding, sigmas, latent, `pred_xstart`, SDF field, four decoder parameter
  tensors, mesh counts, bounding box, vertices, and CPU face identity. The
  smoke mesh had 1,016 vertices and 1,948 faces.
- **Production mesh smoke:** Prompt `a red chair`, seed 7, 64 steps, grid 64,
  guidance 15. CPU and Metal each reported 9,882 vertices, 19,684 faces,
  `boundaryEdgeCount: 0`, `nonManifoldEdgeCount: 0`,
  `invalidFaceCount: 0`, and `isClosed: true`.
- **Fresh download:** The verified private staging snapshot passed checksums,
  L1, and L2 before publication. The fresh public repository was then seeded
  from that payload at `d71f70a09b9c03535188d006a719edda5c19a6eb`, downloaded
  again, and all 1,102 checksum entries reverified.
- **Swift unit suite:** After adding the image-orientation regression test,
  `ShapERuntimeTests` executed 36 tests with 3 skipped and 0 failures.
- **Image conditioning (2026-11-17 follow-up):** A new Torch-oracle harness
  passed on both a deterministic synthetic image and
  `TripoSR/examples/chair.png`. It separately gates Swift CLIP preprocessing,
  `ClipImageEmbedder` grid output from an exact shared input, `Image300M`
  with the Torch grid, `Image300M` with the Swift grid, and a 64-step,
  grid-64 CPU/Metal mesh smoke. Neural component correlations printed as
  1.00000, with mean absolute errors of about `7.1e-6` for the grid and
  `1.0e-6`–`2.4e-6` for Image300M. Both mesh backends produced identical,
  closed counts with zero boundary, nonmanifold, or invalid faces. The run
  found and fixed a vertical flip in `ImageProcessor`. The fix and harness
  are included in this candidate tree and recorded in the Hub summary; the
  Hub evidence/model card were refreshed at
  `d71f70a09b9c03535188d006a719edda5c19a6eb`.
- **Partial download and downloader (2026-11-17 follow-up):**
  `cli.run.download_shap_e_bundle` pins
  `d71f70a09b9c03535188d006a719edda5c19a6eb`, verifies SHA-256 entries, and
  has 7 unit tests for scope/checksum behavior. Its `--text-only` mode
  downloaded 495 files (494 checksum entries) with no `clip_vision` or
  `image300m` directory; that partial bundle then generated a closed
  688-vertex/1,356-face text mesh in CPU and Metal. Text-only mode rejects a
  non-empty existing image family so it cannot mislabel a full local bundle.

### Scope of that evidence

The original fresh-download verification covered the **text-conditioned**
path. The follow-up above now covers Swift image preprocessing, the CLIP
vision grid, Image300M, sampling through decoder projection, and CPU/Metal
mesh topology against upstream Torch components. It does not add a Python MLX
image runtime: Python's typed bundle loader still has no `clip_vision` or
`image300m` artifact loader, and platform image resampling is gated by
correlation/mean-error thresholds rather than exact equality. The current
Hub model card now reports the passed image parity and points to
`validation/image-parity.json` for the exact metrics and mesh counts.

## Publication record

The v0.1.0 gates are complete:

1. The separate public source remote was created and received only the
   candidate's initial commit.
2. An anonymous recursive clean clone passed full local verification,
   downloaded the public bundle at the pinned revision, and generated a
   closed text mesh.
3. Matching `v0.1.0` identifiers were created on GitHub and Hugging Face
   after that verification.

Resolved since the prior audit: the GitHub profile public email is now
`alex@signalnotnoise.dev`.

## TripoSR weights track (2026-11-19)

The separate TripoSR track is closed. `signalNotNoise/triposr-mlx` was
uploaded private, verified, and flipped **public on 2026-11-23** at the
owner's direction; the post-flip anonymous gates below passed the same day.

### Canonical artifact decision

- **`triposr_swift_ready.safetensors` is the canonical published artifact**
  (1,674,793,528 bytes, 499 tensors, SHA-256
  `83563625f39180f505a3ccb855e67b59b65989f6545015672465d7487c4bc2cb`).
  It is the only file `TripoSRModel.loadWeights` accepts: keys and layouts
  already match the Swift modules, and the loader verifies shapes and
  rejects unused keys.
- `triposr_mlx.safetensors` (1,677,170,936 bytes, 549 tensors, SHA-256
  `f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b`) is
  an intermediate pack with upstream PyTorch keys. Its only consumer is
  `TripoSR/convert_weights_to_mlx.py`; it is hashed into
  `provenance.json` and **never published**.
- Conversion is reproducible: `export_tripo_weights.py` pins upstream
  revision `5b521936b01fbe1890f6f9baed0254ab6351c04a` (checkpoint
  1,677,246,742 bytes, SHA-256
  `429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee`),
  `convert_weights_to_mlx.py` fails on any unmapped key, and a full
  re-export on 2026-11-19 reproduced both files byte-for-byte.

### Redistribution terms (recorded conclusion)

Derived-weight redistribution is permitted under the upstream MIT release:

- Upstream `VAST-AI-Research/TripoSR` `LICENSE` is the MIT License,
  Copyright (c) 2024 Tripo AI & Stability AI (carried in the submodule and
  in the bundle as `licenses/TripoSR.txt`).
- The upstream README states: "The model is released under the MIT license,
  which includes the source code, pretrained models, and an interactive
  online demo."
- The Hugging Face model card for `stabilityai/TripoSR` declares
  `license: mit` and "License: MIT", and the repository is not gated
  (checked 2026-11-19).
- The checkpoint's image tokenizer is initialized from
  `facebook/dino-vitb16` per the upstream config; those parameters ship
  inside the upstream checkpoint under the release above and are attributed
  in the bundle `NOTICE` and the repository [`NOTICE`](../NOTICE).

Conversion is a format/layout export, not training or fine-tuning. The MIT
copyright and permission notices are preserved in the bundle.

### Publication record

- Packaged by `scripts/package_triposr_weights.py`; the bundle's
  `provenance.json` records the code and submodule commits
  (`6da4a3cd3f8454f7c98607249476a26c836509ee` for the submodule, the pin
  used here).
- Hub repository `signalNotNoise/triposr-mlx`: weights commit
  `8353300f4a032aca9099c09344292a8809c2eb58` (uploaded private), then
  download-verification metadata commit
  `d0094e6537dea1224b1619ac2780b27bfa129e1a`.
- While private: fresh authenticated download of the pinned revision
  passed all checksums and the mesh test, and anonymous access was
  verified denied (proving privacy).
- **Public flip (2026-11-23):** visibility set to public with the weights
  blob unchanged. An unauthenticated download (fresh `HOME`, no
  `HF_TOKEN`) of pinned revision `d0094e6537dea1224b1619ac2780b27bfa129e1a`
  passed `shasum -a 256 -c checksums.sha256` (10 entries), and the
  downloaded weights passed the RuntimeWrapper mesh test with the
  identical closed topology (2,558 vertices / 5,116 faces). The flip was
  then recorded in `validation/download-verification.json` at commit
  `17a0c5286e2b2bb6fc9e3654ca31cfc9dc4b2041` (current head), whose own
  anonymous download and checksum verification also passed.

### TripoSR evidence at this pin

- **Tensor parity (`gate.triposr`), rerun end-to-end 2026-11-19:** all five
  checks passed (image_tokens corr 0.98644, backbone_out 0.99735,
  triplanes 0.99745, density 0.95021, color 0.99609). High correlation,
  not bit-exact. The Python dump needed a `huggingface-hub>=1.5,<2`
  overlay for transformers 5.5 (recorded in `TripoSR/SCRIPTS.md`).
- **Mesh stage:** the parity gate does not compare meshes. The env-gated
  `TripoSRMeshRuntimeTests` covers that stage end-to-end with the real
  weights: `chair.png` at grid 64 produced a closed mesh (2,558 vertices,
  5,116 faces, zero boundary/non-manifold/invalid faces), finite
  in-bounds vertices, and in-range RGBA colors. A mesh-vs-mesh numerical
  comparison against upstream `torchmcubes` remains undone and is a stated
  limitation, not a passed gate.
- **Unit tests:** `TripoSRRuntimeTests` is no longer a placeholder
  (4 deterministic tests, run via `xcodebuild`), and `verify.sh` full
  mode now runs the RuntimeWrapper suite.

### Remaining TripoSR gates

1. Optional: mesh-vs-mesh comparison against upstream Python meshing, and
   bicubic DINO pos-embed interpolation to tighten parity.
2. Mesquette integration is coordinated separately and is not part of
   this record.

## Reproduce or audit the candidate

The tested export snapshot identifier is recorded in the Hub
`provenance.json`; do not silently substitute a later tree when comparing
against the recorded hashes.

To inspect the exact verified public weights, pin the release revision:

```python
from huggingface_hub import snapshot_download

bundle = snapshot_download(
    "signalNotNoise/shap-e-mlx",
    revision="d71f70a09b9c03535188d006a719edda5c19a6eb",
)
print(bundle)
```

Verify payload files against `checksums.sha256` before using them.

To re-export from upstream checkpoints, use the Hub files
`reproduce_export.py` and `requirements-validation.txt` at the pinned
revision, or call the source exporter in this repository:

```bash
cd shap_e/src
python -m mlx_shap_e.offline_tools.runtime_artifact_exports \
  --bundle-dir ../../runtime_bundle \
  --cache-dir ../../shap_e_model_cache
```

That exporter calls
`prepare_runtime_artifacts`, which writes all seven families. It does not add
the Hub-only root files (`bpe_simple_vocab_16e6.txt.gz`, `provenance.json`,
`checksums.sha256`, licenses, model card, `validation/`, or
`reproduce_export.py`); those belong to release packaging.

## Release procedure for the next bundle

1. Freeze one code commit and record it before exporting.
2. Export all seven families into a new empty staging bundle; never upload a
   long-lived local `runtime_bundle/` without regenerating or proving its
   checkpoint provenance and checksums.
3. Run L0 component comparisons, L1, L2, Swift unit tests, and at least one
   production CPU and Metal mesh topology check. Add image-path gates before
   claiming image validation.
4. Package `provenance.json`, upstream licenses, BPE vocabulary,
   `checksums.sha256`, model card, release requirements, a re-export driver,
   and the `validation/` summaries.
5. Upload to a private Hub revision. Download that exact revision into a
   fresh directory, verify every checksum, then rerun L1/L2 and `ShapECLI`
   from the downloaded files.
6. Update this status file, [File structure](FILE-STRUCTURE.md), the root
   [README](../README.md), the [ShapERuntime README](../shap_e/swift/README.md),
   and the knowledge graph in the same code change.
7. Flip GitHub and Hugging Face only after all publication gates above are
   resolved or explicitly accepted, then repeat the test anonymously from a
   fresh recursive clone and a fresh Hub download.

## Update rule

Any change to a repository URL, visibility, pinned revision, bundle family,
tensor count, byte total, validation scope, or publication gate must update
this file and [File structure](FILE-STRUCTURE.md) together. Do not copy a
“passed” result forward to a new code commit or bundle revision without
rerunning it.
