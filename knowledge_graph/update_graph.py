#!/usr/bin/env python3
"""Rebuild knowledge_graph/graph.json from curated.json plus a source scan.

Hand-edited meaning lives in curated.json. This script owns graph.json and the
generated block in README.md. It is deterministic: a second run with no tree
changes rewrites nothing.

Stdlib only.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = ROOT / "knowledge_graph"
CURATED_PATH = GRAPH_DIR / "curated.json"
GRAPH_PATH = GRAPH_DIR / "graph.json"
README_PATH = GRAPH_DIR / "README.md"

GENERATED_START = "<!-- knowledge-graph:generated:start -->"
GENERATED_END = "<!-- knowledge-graph:generated:end -->"

SOURCE_SUFFIXES = {".py", ".swift", ".metal", ".sh"}
PRUNE_DIRS = {
    ".git",
    ".build",
    ".derivedData",
    ".mlxEnv",
    ".mlx_venv",
    ".mypy_cache",
    ".pytest_cache",
    ".swiftpm",
    ".venv",
    "__pycache__",
    "env",
    "examples",
    "figures",
    "mesh_ab_smoke",
    "Niko",
    "node_modules",
    "parity",
    "runtime_bundle",
    "shap_e_model_cache",
    "triposr_weights",
    "venv",
    "xcuserdata",
}
LOCAL_SWIFT_IMPORTS = {
    "ShapERuntime": "pkg.swift.shap-e",
    "TripoSRRuntime": "pkg.swift.triposr",
    "RuntimeWrapper": "pkg.swift.wrapper",
}
# Longest prefix wins. A scanned file is attached to the package that owns it.
CONTAINERS = (
    ("shap_e/swift/", "pkg.swift.shap-e"),
    ("shap_e/src/", "pkg.py.mlx-shap-e"),
    ("TripoSR/swift/", "pkg.swift.triposr"),
    ("TripoSR/tsr/", "sys.triposr"),
    ("TripoSR/scripts/", "sys.triposr"),
    ("runtime_wrapper/", "pkg.swift.wrapper"),
    ("tests/", "test.py.unit"),
    ("shap_e/scripts/", "sys.shap-e"),
)
IMPORT_ROOTS = (
    ROOT / "shap_e" / "src",
    ROOT / "TripoSR",
    ROOT,
)


def norm(path: str | Path) -> str:
    text = Path(path).as_posix()
    if text.startswith("./"):
        text = text[2:]
    return text


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def dump_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == text:
        return
    path.write_text(text, encoding="utf-8")


def should_prune(directory: Path) -> bool:
    name = directory.name
    if name in PRUNE_DIRS or name.endswith(".egg-info"):
        return True
    return False


def skip_file(path: Path) -> bool:
    posix = rel(path)
    if posix.startswith("shap_e/swift/test_"):
        return True
    if path.name == "update_graph.py":
        return False
    return False


def iter_source_files() -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os_walk(ROOT):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not should_prune(current / name)]
        for name in filenames:
            path = current / name
            if path.suffix not in SOURCE_SUFFIXES and path.name != "Package.swift":
                continue
            if skip_file(path):
                continue
            found.append(path)
    return sorted(found, key=rel)


def os_walk(root: Path):
    import os

    return os.walk(root)


def scan_id(path: Path) -> str:
    return "scan:" + rel(path)


def kind_for(path: Path) -> str:
    posix = rel(path)
    name = path.name
    if "/Tests/" in posix or posix.startswith("tests/") or name.startswith("test_"):
        return "test"
    if name == "Package.swift" or name.endswith(".sh"):
        return "entrypoint"
    return "module"


def python_symbols(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def swift_symbols(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        parts = stripped.split()
        keywords = {"class", "struct", "enum", "protocol", "actor", "func"}
        for index, part in enumerate(parts[:-1]):
            if part in keywords:
                token = parts[index + 1].split("(")[0].split(":")[0].split("<")[0]
                if token.isidentifier() and token not in names:
                    names.append(token)
                break
    return names[:40]


def python_imports(path: Path) -> list[tuple[str | None, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return []
    found: list[tuple[str | None, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, 0))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.append((node.module, node.level or 0))
            else:
                for alias in node.names:
                    found.append((alias.name, node.level or 0))
    return found


def resolve_python(module: str | None, level: int, importer: Path) -> Path | None:
    if level:
        package = importer.parent
        for _ in range(level - 1):
            package = package.parent
        base = package if not module else package.joinpath(*module.split("."))
        if base.is_dir():
            init_candidate = base / "__init__.py"
            return init_candidate if init_candidate.is_file() else None
    elif module:
        base = None
        for root in IMPORT_ROOTS:
            candidate = root.joinpath(*module.split("."))
            if candidate.with_suffix(".py").is_file() or (candidate / "__init__.py").is_file():
                base = candidate
                break
        if base is None:
            return None
    else:
        return None
    file_candidate = base if base.suffix == ".py" else base.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    init_candidate = base / "__init__.py"
    if init_candidate.is_file():
        return init_candidate
    return None


def swift_imports(path: Path) -> list[str]:
    modules: list[str] = []
    if path.suffix != ".swift":
        return modules
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("import ") or stripped.startswith("import class"):
            # `import struct` / `import class` are not module imports.
            if not stripped.startswith("import "):
                continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        if parts[1] in {"class", "struct", "enum", "protocol", "func", "typealias"}:
            if len(parts) < 3:
                continue
            name = parts[2].split(".")[0]
        else:
            name = parts[1].split(".")[0]
        if name not in modules:
            modules.append(name)
    return modules


def container_for(path: Path) -> str | None:
    posix = rel(path)
    match = None
    for prefix, node_id in CONTAINERS:
        if posix.startswith(prefix) and (match is None or len(prefix) > len(match[0])):
            match = (prefix, node_id)
    return None if match is None else match[1]


def edge_key(edge: dict) -> tuple[str, str, str]:
    return (edge["from"], edge["to"], edge["rel"])


def render_generated(curated_count: int, scan_count: int, edge_count: int, uncurated: int, missing: int) -> str:
    return (
        f"{GENERATED_START}\n"
        f"{curated_count} curated nodes, {scan_count} scanned nodes, {edge_count} edges. "
        f"{uncurated} source files have no curated node. "
        f"{missing} curated paths are missing on disk.\n"
        "Regenerate with `python3 knowledge_graph/update_graph.py`. "
        "The pre-commit hook runs that command and stages the result.\n"
        f"{GENERATED_END}"
    )


def update_readme(block: str) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    start = text.find(GENERATED_START)
    end = text.find(GENERATED_END)
    if start == -1 or end == -1 or end < start:
        raise SystemExit(f"{README_PATH} is missing the generated-block markers.")
    end += len(GENERATED_END)
    updated = text[:start] + block + text[end:]
    if updated != text:
        README_PATH.write_text(updated, encoding="utf-8")


def build() -> dict:
    if not CURATED_PATH.is_file():
        raise SystemExit(
            "knowledge_graph/curated.json is missing. Restore it from git. "
            "Do not rebuild it from graph.json; graph.json is generated."
        )
    curated = load_json(CURATED_PATH)
    nodes: list[dict] = []
    path_to_id: dict[str, str] = {}
    node_ids: set[str] = set()
    for node in curated["nodes"]:
        copied = dict(node)
        copied["origin"] = "curated"
        if copied.get("path"):
            copied["path"] = norm(copied["path"])
            # Optional paths name ignored local artifacts (downloaded
            # checkpoints and converted bundles). Their presence varies by
            # machine, so they must not make graph generation nondeterministic
            # between a developer checkout and a clean CI clone.
            if not copied.get("optional") and not (ROOT / copied["path"]).exists():
                copied["path_missing"] = True
            path_to_id[copied["path"]] = copied["id"]
        nodes.append(copied)
        node_ids.add(copied["id"])

    edges: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in curated["edges"]:
        copied = dict(edge)
        copied["origin"] = "curated"
        edges.append(copied)
        seen.add(edge_key(copied))

    scan_errors: list[str] = []
    uncurated: list[str] = []
    source_files = iter_source_files()
    for path in source_files:
        posix = rel(path)
        if posix in path_to_id:
            continue
        node_id = scan_id(path)
        if node_id in node_ids:
            scan_errors.append(f"duplicate scan id {node_id}")
            continue
        symbols: list[str] = []
        if path.suffix == ".py":
            symbols = python_symbols(path)
        elif path.suffix == ".swift":
            symbols = swift_symbols(path)
        scanned = {
            "id": node_id,
            "kind": kind_for(path),
            "name": path.name,
            "path": posix,
            "origin": "scan",
            "summary": "Source file discovered by the filesystem scan. It has no curated summary yet.",
        }
        if symbols:
            scanned["symbols"] = symbols
        nodes.append(scanned)
        node_ids.add(node_id)
        path_to_id[posix] = node_id
        uncurated.append(posix)
        parent = container_for(path)
        if parent and parent in node_ids:
            edge = {"from": parent, "to": node_id, "rel": "contains", "origin": "scan"}
            if edge_key(edge) not in seen:
                edges.append(edge)
                seen.add(edge_key(edge))

    for path in source_files:
        posix = rel(path)
        owner = path_to_id.get(posix)
        if owner is None:
            continue
        if path.suffix == ".py":
            for module, level in python_imports(path):
                try:
                    target = resolve_python(module, level, path)
                except OSError as exc:
                    scan_errors.append(f"{posix}: {exc}")
                    continue
                if target is None or not target.is_file():
                    continue
                try:
                    target_posix = rel(target)
                except ValueError:
                    continue
                target_id = path_to_id.get(target_posix)
                if target_id is None or target_id == owner:
                    continue
                edge = {"from": owner, "to": target_id, "rel": "calls", "origin": "scan"}
                if edge_key(edge) not in seen:
                    edges.append(edge)
                    seen.add(edge_key(edge))
        elif path.suffix == ".swift":
            for module in swift_imports(path):
                target_id = LOCAL_SWIFT_IMPORTS.get(module)
                if target_id is None or target_id not in node_ids or target_id == owner:
                    continue
                edge = {"from": owner, "to": target_id, "rel": "depends_on", "origin": "scan"}
                if edge_key(edge) not in seen:
                    edges.append(edge)
                    seen.add(edge_key(edge))

    missing = [node["path"] for node in nodes if node.get("path_missing")]
    curated_count = sum(1 for node in nodes if node["origin"] == "curated")
    scan_count = sum(1 for node in nodes if node["origin"] == "scan")
    graph = {
        "name": curated.get("name", "apple-mlx-foundations"),
        "description": curated.get("description", ""),
        "schema": curated["schema"],
        "meta": {
            "generator": "knowledge_graph/update_graph.py",
            "curated_source": "knowledge_graph/curated.json",
            "curated_nodes": curated_count,
            "scanned_nodes": scan_count,
            "edges": len(edges),
            "uncurated_files": uncurated,
            "missing_curated_paths": missing,
            "scan_errors": scan_errors,
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def main() -> int:
    check_only = "--check" in sys.argv
    graph = build()
    block = render_generated(
        graph["meta"]["curated_nodes"],
        graph["meta"]["scanned_nodes"],
        graph["meta"]["edges"],
        len(graph["meta"]["uncurated_files"]),
        len(graph["meta"]["missing_curated_paths"]),
    )
    if check_only:
        current_graph = GRAPH_PATH.read_text(encoding="utf-8") if GRAPH_PATH.is_file() else ""
        expected_graph = json.dumps(graph, indent=2, ensure_ascii=False) + "\n"
        readme = README_PATH.read_text(encoding="utf-8")
        start = readme.find(GENERATED_START)
        end = readme.find(GENERATED_END)
        current_block = readme[start : end + len(GENERATED_END)] if start != -1 and end != -1 else ""
        if current_graph != expected_graph or current_block != block:
            print("knowledge graph is stale. Run: python3 knowledge_graph/update_graph.py", file=sys.stderr)
            return 1
        print("knowledge graph is current")
        return 0
    dump_json(GRAPH_PATH, graph)
    update_readme(block)
    meta = graph["meta"]
    print(
        f"knowledge graph updated: {meta['curated_nodes']} curated, "
        f"{meta['scanned_nodes']} scanned, {meta['edges']} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
