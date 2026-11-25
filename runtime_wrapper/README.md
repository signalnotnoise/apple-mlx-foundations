# RuntimeWrapper

`RuntimeWrapper` is the optional model-selection layer for the independent Swift runtimes:

```text
RuntimeWrapper
├── ShapERuntime      text or image → mesh
└── TripoSRRuntime    image → density/color
```

The wrapper converts either backend into its model-neutral `ThreeDMeshResult`, backed by
the shared `MeshResult`, and reuses Shap-E's CPU or Metal marching-cubes implementation. Neither
model runtime depends on the other; applications that only need one model can import that
model package directly.

## Checkout

`RuntimeWrapper` depends on the `TripoSR` submodule. From the repository root:

```bash
git submodule sync
git submodule update --init
```

(`sync` refreshes a cached submodule URL from `.gitmodules` — needed in
clones made before the TripoSR submodule switched from SSH to HTTPS.)

## Usage

```swift
import RuntimeWrapper

let runtime = try ThreeDGenerationRuntime(
    configuration: .tripoSR(weightsURL: tripoSRWeightsURL)
)
let meshes = try await runtime.generate(image: cgImage)
```

For Shap-E:

```swift
let runtime = try ThreeDGenerationRuntime(
    configuration: .shapE(
        modelBundleURL: modelBundleURL,
        bpeVocabularyURL: bpeVocabularyURL
    )
)
```

Shap-E accepts image or text input. TripoSR accepts image input only.

The TripoSR backend loads `triposr_swift_ready.safetensors` (the canonical
converted TripoSR weights; see `TripoSR/SCRIPTS.md`).

## Tests

```bash
cd runtime_wrapper && swift test
```

`TripoSRMeshRuntimeTests` additionally runs a real-weights end-to-end mesh
test when both variables point at existing files (otherwise it skips; a
skip is not a pass):

```bash
TRIPOSR_WEIGHTS_PATH=../TripoSR/triposr_weights/triposr_swift_ready.safetensors \
TRIPOSR_IMAGE_PATH=../TripoSR/examples/chair.png \
swift test --filter TripoSRMeshRuntimeTests
```
