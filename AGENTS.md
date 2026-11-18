# Repository rules

## Coding standards

The detailed standards are in
[`docs/CODING-STANDARDS.md`](docs/CODING-STANDARDS.md). They are mandatory for
humans and coding agents working in this repository.

Non-negotiable summary:

- Inspect the relevant code path, callers, tests, and conventions before
  editing; fix root causes with the smallest maintainable change.
- Favor simple, explicit, modular code; do not add abstractions or
  dependencies without a current, demonstrated need.
- Each production source file has one primary class/type and one clear
  responsibility. Cohesive supporting error/DTO types require clear
  justification. There is **no file line-count limit**.
- Document public APIs, nontrivial behavior, invariants, assumptions, edge
  cases, and limitations. Update docs in the same change as behavior.
- Add or update deterministic tests for behavior changes and regression
  tests for bugs. Never weaken a real test to obtain green.
- Validate inputs, handle errors explicitly, preserve security boundaries,
  and never hardcode secrets or personal machine paths.
- Never claim success without observed evidence; report anything skipped or
  unverified.

Before finishing a code or documentation change, run the applicable checks,
including:

```bash
python3 scripts/check_standards.py --changed
./scripts/verify.sh --quick
```

Run `./scripts/verify.sh` for Swift/runtime changes unless the focused
evidence and any limitation are explicitly reported. The checker cannot judge
simplicity or documentation quality; reviewers remain responsible for those.

## Knowledge graph

`knowledge_graph/graph.json` is the map of this repo. It has to match the tree after every change, including a change made by a different tool.

Before you finish any turn that creates, edits, deletes, or moves a file here, run from the repo root:

```bash
python3 knowledge_graph/update_graph.py
```

The task is not done until that command has run.

The script merges `knowledge_graph/curated.json` with a scan of the source tree and rewrites `knowledge_graph/graph.json` plus the generated block in `knowledge_graph/README.md`. Commit those outputs with the change. `.githooks/pre-commit` runs the same command and stages the outputs, so a commit from any tool still refreshes the graph. Do not bypass the hook with `--no-verify`.

If `git config --get core.hooksPath` is not `.githooks`, run `knowledge_graph/install_hooks.sh` once for this clone.

Edit `knowledge_graph/curated.json` in the same change when you:

- add, remove, or rewire a pipeline stage, package, artifact, parity gate, or public runtime API
- change what a curated module does
- fix or retire a note

A new source file with no curated node is still recorded by the scan (`meta.uncurated_files`). Give it a curated node when it becomes part of the architecture. Delete curated nodes whose files you removed.

Do not hand-edit `graph.json`. The script owns that file.
