# Shap-E mesh hole diagnosis (CPU vs Metal A/B)

When Vision shows **holes / white melts** on generated meshes, separate:

1. **Shap-E model / SDF quality** (Python would look the same)  
2. **Swift Metal marching cubes bug** (only GPU extract is holey)  
3. **Live partials** (mid-run previews — not final meshes)

This harness extracts the **SDF once**, then runs **CPU** and **Metal** MC on that exact field, exports OBJs, and counts **boundary edges** (open holes/cracks).

---

## Quick run

```bash
# repo root — needs runtime_bundle/ + .mlx_venv CLIP BPE
./shap_e/scripts/run_mesh_ab.sh

# Match Vision-ish settings (slower). Default CFG uncond = embed("") (production generate).
./shap_e/scripts/run_mesh_ab.sh \
  --prompt "a ford mustang" \
  --seed 859729 \
  --steps 64 \
  --grid 128 \
  --guidance 15
```

Or after building:

```bash
cd shap_e/swift
xcodebuild -scheme ShapECLI -destination 'platform=macOS' \
  -derivedDataPath ./.derivedData build

# from repo root:
./shap_e/swift/.derivedData/Build/Products/Debug/ShapECLI --ab-mesh \
  --prompt "a ford mustang" --seed 859729 --steps 64 --grid 128
```

**Do not** use plain `swift run` if MLX complains about `default.metallib` — use **xcodebuild**.

---

## Outputs (`shap_e/mesh_ab/` by default)

| File | Meaning |
|------|---------|
| `sdf_field.npy` | Unpadded SDF `[G,G,G]` shared by both MC paths |
| `mesh_cpu.obj` | CPU `MarchingCubes` |
| `mesh_metal.obj` | Metal `MetalMarchingCubes` |
| `ab_report.json` | Counts, boundary edges, timings, sdf min/max |

---

## How to read the VERDICT

Printed summary uses `MeshTopology`:

| Result | Interpretation |
|--------|----------------|
| **Both closed** (`boundary_edges=0`) | Topology OK. White “holes” are likely **color**, shading, or **melted SDF** still triangulated |
| **Both open** (`boundary_edges>0`) | **SDF / model** incomplete for this prompt/seed — not Metal-only |
| **Metal open, CPU closed** | **Metal MC / weld / SDF host buffer** bug |
| **CPU open, Metal closed** | Unexpected; treat as CPU MC bug |
| **Metal FAILED** | GPU hang / metallib / kernel error — CPU mesh still usable for model quality |

Also open both OBJs in Blender:

1. Wireframe  
2. Select → Select All by Trait → **Non Manifold** / boundary  
3. Compare face counts from the console / `ab_report.json`

---

## Recommended debug settings

| Setting | Debug | Vision “High” (slow / noisy) |
|---------|-------|------------------------------|
| steps | 32–64 | 128–164 |
| grid | 64–128 | 250 |
| live partials | **off** (in app) | on — do not trust partials |
| batch | 1 | 2–4 |

Cars (`ford mustang`) are a hard class for classic Shap-E; expect melts even when topology is closed.

---

## Code map

| Piece | Path |
|-------|------|
| CLI mode | `shap_e/swift/Sources/ShapECLI/main.swift` (`--ab-mesh`) |
| Same-SDF extract | `STFMeshRenderer.meshFromSDFField` |
| Boundary metrics | `MeshTopology.swift` |
| OBJ export | `MeshExport.swift` |
| Shell | `shap_e/scripts/run_mesh_ab.sh` |

Related: MLX↔Swift tensor parity is **`./shap_e/scripts/run_parity.sh`** (different job).

---

## Acceptance criteria (what “good” looks like)

You can trust the Swift mesh path when:

1. On a fixed seed, **CPU and Metal** agree on closedness (both closed or both similarly open).  
2. Open boundaries on **both** match **Python** mesh quality for the same seed (model limit).  
3. Metal-only open boundaries are treated as **bugs**, not “just Shap-E”.
