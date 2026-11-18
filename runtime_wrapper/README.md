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
git submodule update --init
```

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
