#!/bin/sh
# Point this clone at the committed hooks. Run once per clone.
root=$(git rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit knowledge_graph/update_graph.py knowledge_graph/install_hooks.sh
echo "git hooks path set to .githooks"
