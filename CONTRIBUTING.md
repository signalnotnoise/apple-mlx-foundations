# Contributing

Follow [`AGENTS.md`](AGENTS.md) and
[`docs/CODING-STANDARDS.md`](docs/CODING-STANDARDS.md). Those rules apply to
human and agent-authored changes.

## One-time setup

```bash
git clone --recurse-submodules https://github.com/signalnotnoise/apple-mlx-foundations.git
cd apple-mlx-foundations
knowledge_graph/install_hooks.sh
python3 -m venv .mlx_venv
source .mlx_venv/bin/activate
pip install -e '.[dev]'
```

The pre-commit hook runs the staged coding-standards checker and refreshes
the knowledge graph. Do not bypass it with `--no-verify`.

## Before opening a pull request

Inspect the relevant implementation and tests first, make the smallest
cohesive change, update documentation, and run:

```bash
python3 scripts/check_standards.py --changed
./scripts/verify.sh --quick
python3 knowledge_graph/update_graph.py
```

Run `./scripts/verify.sh` for Swift/runtime changes, plus the parity harness
that covers the touched model path. The pull-request template records exact
commands, outcomes, skipped checks, risks, and any standards exception.

There is no file line-count rule. Review instead asks whether each file has
one primary type/responsibility and whether supporting types are cohesive.
Existing multi-type files are a baseline: their primary-type count must not
grow without a recorded, reviewed exception.
