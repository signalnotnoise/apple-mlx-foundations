# Shap-E Workspace Notes

This folder contains the MLX Shap-E runtime and tooling under:

- `shap_e/src/mlx_shap_e`
- `shap_e/src/cli`
- `shap_e/swift` (the Swift port)

The complete repository/generated-artifact map and runtime-bundle schema are
in [`docs/FILE-STRUCTURE.md`](../docs/FILE-STRUCTURE.md). Release revisions and
validation scope are in [`docs/RELEASE-STATUS.md`](../docs/RELEASE-STATUS.md).

## Run Convention

Run scripts as modules from `shap_e/src` (paths below are repo-root relative):

```bash
cd shap_e/src
python -m cli.run.sample_text300m_karras_mlx --help
```

## Key Commands

Export the complete seven-family runtime bundle:

```bash
python -m mlx_shap_e.offline_tools.runtime_artifact_exports \
  --bundle-dir ../../runtime_bundle \
  --cache-dir ../../shap_e_model_cache
```

This calls `prepare_runtime_artifacts`, whose per-family writers live in
`mlx_shap_e.offline_tools.runtime_artifact_exports`. The commands under
`cli.export.*` are narrower wrappers/utilities; do not treat the two examples
previously shown there as a complete bundle export.

Run end-to-end pipeline:

```bash
python -m cli.run.run_text300m_mlx_pipeline --prompt "a red chair"
```

Run parity checks:

```bash
python -m cli.check.check_mlx_diffusion_step_parity --prompt "a red chair"
python -m cli.check.check_mlx_karras_loop_parity --prompt "a red chair" --steps 64
python -m cli.check.check_mlx_karras_model_output_parity --prompt "a red chair" --steps 64
```

For broader project setup and command examples, see the root [`README.md`](../README.md).
