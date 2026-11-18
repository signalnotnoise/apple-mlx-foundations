## Summary

What problem does this change solve, and what is the root cause if it is a fix?

## Scope and design

- [ ] I inspected the relevant code path, callers, tests, and conventions.
- [ ] This is the smallest maintainable change; no unrelated work is included.
- [ ] New/changed files each have one clear responsibility and one primary
      class/type, or the exception and reason are recorded below.
- [ ] New abstractions/dependencies are justified below, or none were added.

## Documentation

- [ ] Public APIs and nontrivial behavior are documented.
- [ ] Affected project documentation is updated.
- [ ] `knowledge_graph/curated.json` is updated if architecture, a pipeline
      stage, artifact, parity gate, or public runtime API changed.
- [ ] `python3 knowledge_graph/update_graph.py` was run.

## Verification evidence

List exact commands and observed outcomes. Do not replace results with
“tests pass” when a test was skipped or unavailable.

- [ ] `python3 scripts/check_standards.py --changed`
- [ ] Relevant focused tests:
- [ ] Relevant builds/parity gates:
- [ ] Existing-behavior/regression checks:

## Risk and uncertainty

- Security, concurrency, lifecycle, or compatibility risks:
- Anything intentionally not verified:
- Standards exceptions (file, rule, reason, review date):
