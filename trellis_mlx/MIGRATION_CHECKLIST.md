# TRELLIS → MLX migration checklist

Status: **not started** (workspace scaffold only). Check items only with
recorded evidence (command + output artifact), mirroring the Shap-E gates.

## Gate 0 — Reference baseline

- [ ] Run upstream `TRELLIS` image-to-3D inference locally (Torch/MPS or CPU)
      on a fixed test image; record seed, versions, and output mesh stats.
- [ ] Freeze the exact upstream commit and HF weight revision used as oracle.
- [ ] Dump per-stage oracle tensors: DINOv2 features, SS latents, decoded
      voxel coords, SLat latents, final vertices/faces/colors.

## Gate 1 — Weight export

- [ ] Export fp16 checkpoints (`ss_flow`, `ss_dec`, `slat_flow`,
      `slat_dec_mesh`, DINOv2) into a versioned runtime-bundle layout with
      manifest + SHA-256 checksums + provenance (upstream repo/revision).
- [ ] Round-trip check: reloaded tensors equal the safetensors sources.

## Gate 2 — Python MLX parity (per module, L0)

- [ ] DINOv2 ViT-L/14-reg forward vs oracle features.
- [ ] Sparse-structure DiT block vs oracle (dense reformulation documented).
- [ ] SS conv3d decoder vs oracle occupancy/logits.
- [ ] SLat DiT block (sparse attention reformulation) vs oracle.
- [ ] SLat mesh decoder (Swin windows) vs oracle vertices/faces.
- [ ] FlowEuler guidance-interval sampler trajectory vs oracle sigmas.

## Gate 3 — Python MLX end-to-end (L1)

- [ ] Full image → mesh run in MLX; mesh counts/topology (closed, boundary
      edges) and bbox match oracle within recorded tolerances.

## Gate 4 — Swift runtime (L2)

- [ ] SwiftPM package (`trellis_mlx/swift`) loads the exported bundle.
- [ ] Per-stage Swift vs Python-MLX tensor parity on the frozen inputs.
- [ ] End-to-end Swift mesh smoke (CPU + Metal paths where they diverge).

## Gate 5 — Publication

- [ ] Private Hub upload of converted bundle; anonymous fresh-download
      verification at the pinned revision (checksums + one mesh).
- [ ] Release record updated (`docs/RELEASE-STATUS.md`); only then flip
      Hub repo public and tag.
