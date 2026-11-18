# Coding standards

This document expands the mandatory summary in [`AGENTS.md`](../AGENTS.md).
If the two disagree, `AGENTS.md` is the entry point and this document is the
detailed interpretation; update both in the same change.

## Philosophy

Write simple, modular, well-documented, maintainable code. Avoid unnecessary
complexity, oversized responsibilities, and changes that cannot be verified.

There is deliberately **no file line-count limit**. Size may prompt a design
review, but line count alone is never a pass/fail rule and must not cause
artificial file splitting.

## 1. Simplicity and maintainability

- Favor the simplest correct solution.
- Do not overengineer or add architectural complexity without a demonstrated
  need.
- Do not introduce an abstraction unless it solves a real current problem.
- Avoid unnecessary dependencies, frameworks, and design patterns.
- Prefer readable, explicit code over clever or condensed implementations.
- Do not prematurely optimize; optimize from measurements when performance is
  part of the requirement.
- Follow existing project conventions unless there is a justified reason to
  change them.
- Examine the entire relevant code path before changing shared, concurrent,
  asynchronous, or stateful behavior.

## 2. File and type organization

- Each source file has **one primary class/type and one clear
  responsibility**.
- Small, cohesive supporting error, configuration, result, or DTO types may
  share a file when that makes the primary API clearer; record the reason in
  code documentation or the change description when it is not obvious.
- Break large implementations into cohesive components when responsibilities
  are actually separable. Do not create files merely to satisfy a metric.
- Organize by responsibility and ownership, not by arbitrary fragments of a
  single algorithm.
- Existing multi-type files are legacy baseline, not permission to keep
  adding unrelated primary types. A changed existing file must not increase
  its primary-type count without a recorded exception.

For checker purposes, a “primary type” is a top-level Swift
`class`/`struct`/`enum`/`protocol`/`actor`, or a top-level Python `class`.
Nested types belong to their containing primary type.

## 3. Documentation

Documentation is part of the change, not follow-up work:

- Document all public classes, protocols/interfaces, and exported types.
- Document nontrivial public functions and methods, including parameters,
  return values, errors, side effects, and important contracts.
- Explain complex algorithms, numerical conventions, ordering, coordinate
  systems, units, and non-obvious decisions.
- Document important assumptions, edge cases, and limitations.
- Update user-facing and architecture documentation whenever behavior,
  layout, artifacts, or public APIs change.
- Keep comments accurate; remove or rewrite stale comments.
- Do not narrate obvious code. Explain **why** and what invariants hold.

## 4. Understand before modifying

- Inspect relevant source, callers, tests, data models, and documentation.
- Identify the root cause before implementing a fix.
- Reuse existing functionality and conventions instead of duplicating them.
- Preserve established behavior unless the request explicitly changes it.
- Make the smallest change that correctly and completely addresses the task,
  including its error and edge cases.
- Avoid unrelated modifications.

## 5. Testing and verification

- Test before declaring a change complete.
- Add deterministic tests for new behavior and regression tests for bugs.
- Cover normal behavior, boundaries, invalid inputs, ordering where relevant,
  and failure paths in proportion to risk.
- Run the relevant builds, suites, parity gates, and static checks.
- Never remove, skip, or weaken a real test merely to obtain a green result.
- If a test expectation is wrong, establish that from the governing contract
  or independent oracle before changing it.
- Never claim success without observed evidence. Mark anything not run as
  unverified and explain why.
- A skipped test is not a passing test.

## 6. Debugging

- Reproduce and diagnose before changing code.
- Fix underlying causes rather than masking symptoms.
- Avoid speculative fixes, arbitrary retries/delays, and unexplained
  workarounds.
- Use logs, failing inputs, and reproducible cases as evidence.
- Verify the original failure after the fix and check for regressions.

## 7. Security and reliability

- Validate external and cross-boundary inputs.
- Preserve authentication and authorization boundaries.
- Never hardcode secrets, credentials, tokens, or personal machine paths.
- Handle errors explicitly; do not silently swallow failures.
- Consider ownership, cleanup, cancellation, concurrency, and lifecycle.
- Fail closed for security-sensitive classification and validation.
- Do not introduce a security regression for convenience.

## 8. Required workflow

1. **Inspect** — understand the requirement, architecture, callers, tests,
   and existing conventions.
2. **Plan** — identify the smallest maintainable change and its risks.
3. **Implement** — make focused, modular changes.
4. **Document** — update code comments and affected project docs/graph.
5. **Verify** — run the standards checker, relevant tests/builds, and the
   knowledge-graph refresh.
6. **Report** — state what changed, exact evidence observed, and all
   remaining uncertainty.

## Definition of done

A task is complete only when:

- Requested behavior, including named error/edge cases, is implemented.
- The design is simple, cohesive, and maintainable.
- File/type organization follows Section 2 or has a justified exception.
- Documentation is current.
- Relevant tests/builds pass, or a limitation is explicitly reported.
- Existing behavior is preserved where required.
- No unrelated change is included.
- The final report distinguishes observed results from inference.

## Enforcement model

Policy text alone is not enforcement. This repository uses layered checks:

| Layer | Mechanism | Role |
|---|---|---|
| Agent instructions | `AGENTS.md`, with tool-specific pointers | Tell humans and agents the required workflow before work starts |
| Local standards checker | `scripts/check_standards.py` | Deterministic changed-file checks for primary types, new public API docs, secrets/paths, oversized binaries, and Markdown links |
| Formatters/linters | Ruff and Xcode's `swift-format` where available | Mechanical style and common defects |
| Pre-commit hook | `.githooks/pre-commit` | Fast local feedback plus knowledge-graph refresh |
| Unified verification | `scripts/verify.sh` | Standards, graph, Python tests, Swift tests, and wrapper build |
| Local full verification | `scripts/verify.sh` on a Mac/Xcode host | Python tests, macOS Swift tests, RuntimeWrapper build, and iOS/visionOS compile checks |
| Review | Pull-request template and human/agent review | Simplicity, root cause, design, documentation quality, and risk judgment |

This repository deliberately does **not** use GitHub Actions. Its code,
MLX/Metal paths, model artifacts, and Apple SDK destinations are specific to
macOS, iOS, and visionOS, so verification is run locally on the Mac/Xcode host
that has the required SDKs and weights. Local hooks give fast feedback and
`scripts/verify.sh` is the authoritative local gate; pull-request evidence
records its exact output. Hooks can be absent on a new clone, so setup and
review must confirm `core.hooksPath=.githooks`. Judgment-based rules are
reviewed through the pull-request evidence checklist; no checker claims to
prove a design is simple or a comment is meaningful.

### Baseline and anti-ratchet rule

The existing tree contains multi-type files and undocumented legacy public
declarations. They are a baseline to improve opportunistically, not a reason
for a disruptive mass refactor:

- A **new** production Swift/Python file may declare only one primary type.
- A changed existing production file may not increase its primary-type count.
- A **newly added** public Swift type/function must have a documentation
  comment. A newly added public Python class/function/method in
  `mlx_shap_e` must have a docstring.
- Existing declarations become subject to documentation when their signature
  or behavior is materially changed; reviewers should require the update even
  when the mechanical new-declaration detector cannot prove it.
- `--all` mode reports the legacy inventory as warnings so it remains visible
  without blocking unrelated work.

### Exceptions

Use an exception only when splitting or documenting in the normal way would
make the code less clear. Put one of these comments in the first 30 lines of
the file, including a concrete reason:

```text
standards-exception: primary-type <reason>
standards-exception: public-docs <reason>
```

Exceptions never apply to secrets, personal absolute paths, oversized binary
files, broken documentation links, or failing tests. Reviewers may reject an
exception whose reason is merely convenience.

### Commands

```bash
# Changed-file standards (also run by the pre-commit hook in --staged mode)
python3 scripts/check_standards.py --changed

# Inventory legacy warnings without failing on the baseline
python3 scripts/check_standards.py --all

# Full local verification: root Python tests, macOS Swift tests,
# RuntimeWrapper build, and iOS/visionOS compile checks
./scripts/verify.sh

# Fast verification without Swift suites
./scripts/verify.sh --quick
```

Every repository change still requires:

```bash
python3 knowledge_graph/update_graph.py
```

Install hooks once per clone with `knowledge_graph/install_hooks.sh` (it sets
`core.hooksPath=.githooks`). Do not bypass hooks with `--no-verify`.

### Project-specific verification lanes

- Changes under `shap_e/swift` require the relevant `ShapERuntimeTests` and,
  for runtime behavior, the applicable parity harness.
- Changes under `runtime_wrapper` require its tests/build and the affected
  model package tests.
- Changes under `TripoSR` belong to the submodule repository and require that
  repository's tests before this repository pins a new submodule commit.
- Weight-format or exporter changes require bundle regeneration plus the
  parity/download checks described in `docs/RELEASE-STATUS.md`.
- Documentation-only changes require link/count/path verification and the
  knowledge-graph check; they do not inherit model-validation claims.
