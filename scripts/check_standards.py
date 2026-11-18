#!/usr/bin/env python3
"""Deterministic changed-file enforcement for docs/CODING-STANDARDS.md.

The checker intentionally does not enforce a file line-count limit. It checks
only properties that can be decided mechanically:

- one primary top-level type in new production Swift/Python files;
- no increase in primary-type count in changed existing production files;
- documentation on newly added public Swift/Python declarations;
- secret-shaped strings and personal absolute paths/emails;
- oversized changed binary/source files;
- relative Markdown links in changed Markdown files.

Use --all for a legacy inventory. Existing multi-type/public-doc findings are
warnings in that mode; changed-file mode is an anti-ratchet.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


PRODUCTION_PREFIXES = (
    "shap_e/swift/Sources/",
    "runtime_wrapper/Sources/",
    "shap_e/src/mlx_shap_e/",
    "shap_e/src/cli/",
    "scripts/",
)
SOURCE_SUFFIXES = {".py", ".swift"}
TEXT_SUFFIXES = {
    ".py",
    ".swift",
    ".metal",
    ".sh",
    ".md",
    ".json",
    ".toml",
    ".yml",
    ".yaml",
    ".txt",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_EMAILS = {
    "alex@signalnotnoise.dev",
    "git@github.com",  # SSH transport identity in clone URLs, not a contact
    "noreply@github.com",
}

SECRET_PATTERNS = (
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HOME_PATH_RE = re.compile(r"/" + r"Users/[^/\s]+/")
SWIFT_TYPE_RE = re.compile(
    r"^(?:@\w+(?:\([^)]*\))?\s+)?"
    r"(?:(?:public|open|internal|private|fileprivate|final|indirect)\s+)*"
    r"(?:class|struct|enum|protocol|actor)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
SWIFT_PUBLIC_RE = re.compile(
    r"^(?P<indent>\s*)(?:@\w+(?:\([^)]*\))?\s+)*"
    r"(?:public|open)\s+"
    r"(?:(?:final|indirect|static|class)\s+)*"
    r"(?P<kind>class|struct|enum|protocol|actor|func|init|typealias)\b"
    r"(?:\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*))?"
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" or "warning"
    path: str
    line: int | None
    message: str

    def render(self) -> str:
        location = self.path if self.line is None else f"{self.path}:{self.line}"
        return f"{self.severity.upper()} {location}: {self.message}"


def run_git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def is_production_source(path: str) -> bool:
    if "/Tests/" in path or path.startswith("tests/"):
        return False
    return path.startswith(PRODUCTION_PREFIXES) and Path(path).suffix in SOURCE_SUFFIXES


def has_exception(text: str, kind: str) -> bool:
    """Return whether an exception with a non-empty reason is recorded."""

    pattern = re.compile(
        rf"standards-exception:\s*{re.escape(kind)}\s+\S.*", re.IGNORECASE
    )
    return any(pattern.search(line) for line in text.splitlines()[:30])


def swift_primary_types(text: str) -> list[str]:
    """Return top-level Swift type names in declaration order."""

    names: list[str] = []
    for line in text.splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("//"):
            continue
        match = SWIFT_TYPE_RE.match(line)
        if match and match.group(1) not in {"func", "var", "let"}:
            names.append(match.group(1))
    return names


def python_primary_classes(text: str) -> list[str]:
    """Return top-level Python class names, or an empty list if unparsable."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def primary_types(path: str, text: str) -> list[str]:
    if path.endswith(".swift"):
        return swift_primary_types(text)
    if path.endswith(".py"):
        return python_primary_classes(text)
    return []


def swift_public_declarations(text: str) -> list[tuple[str, int, str]]:
    """Return `(identity, line, name)` for public Swift types/functions."""

    lines = text.splitlines()
    declarations: list[tuple[str, int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = SWIFT_PUBLIC_RE.match(line)
        if not match:
            continue
        kind = match.group("kind")
        name = match.group("name") or kind
        identity = " ".join(line.strip().split())
        declarations.append((identity, index, name))
    return declarations


def swift_declaration_is_documented(lines: list[str], declaration_line: int) -> bool:
    """Whether a declaration has an adjacent Swift documentation comment."""

    cursor = declaration_line - 2
    saw_attribute = False
    while cursor >= 0:
        stripped = lines[cursor].strip()
        if stripped.startswith("///"):
            return True
        if stripped.startswith("@"):
            saw_attribute = True
            cursor -= 1
            continue
        if stripped == "" and not saw_attribute:
            return False
        return False
    return False


def python_public_declarations(text: str) -> dict[str, tuple[int, bool]]:
    """Map qualified public Python names to `(line, has_docstring)`."""

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    found: dict[str, tuple[int, bool]] = {}

    def public(name: str) -> bool:
        return not name.startswith("_")

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef) and public(node.name):
                qualified = f"{prefix}{node.name}"
                found[qualified] = (
                    node.lineno,
                    ast.get_docstring(node) is not None,
                )
                visit(node.body, f"{qualified}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public(node.name):
                qualified = f"{prefix}{node.name}"
                found[qualified] = (
                    node.lineno,
                    ast.get_docstring(node) is not None,
                )

    visit(tree.body)
    return found


def find_secret_issues(path: str, text: str) -> list[Issue]:
    """Find secret-shaped values and personal machine identifiers."""

    issues: list[Issue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if any(pattern.search(line) for pattern in SECRET_PATTERNS):
            issues.append(
                Issue("error", path, line_number, "secret-shaped value must not be committed")
            )
        if HOME_PATH_RE.search(line):
            issues.append(
                Issue(
                    "error",
                    path,
                    line_number,
                    "personal absolute home path must not be committed",
                )
            )
        # Markdown may legitimately quote an old account identifier in an
        # audit/release record. Source and configuration may not embed one.
        emails = [] if path.endswith(".md") else EMAIL_RE.findall(line)
        for email in emails:
            if email.lower() not in ALLOWED_EMAILS and not email.endswith(
                ("@example.com", "@example.org")
            ):
                issues.append(
                    Issue(
                        "error",
                        path,
                        line_number,
                        f"personal or unapproved email address: {email}",
                    )
                )
    return issues


def find_markdown_link_issues(
    root: Path,
    path: str,
    text: str,
    path_exists: Callable[[Path], bool] | None = None,
) -> list[Issue]:
    """Validate relative Markdown destinations in one document."""

    issues: list[Issue] = []
    source = root / path
    for line_number, line in enumerate(text.splitlines(), start=1):
        for raw in MARKDOWN_LINK_RE.findall(line):
            target = raw.strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            destination = target.split("#", 1)[0]
            if not destination:
                continue
            resolved = (source.parent / destination).resolve()
            exists = path_exists(resolved) if path_exists else resolved.exists()
            if not exists:
                issues.append(
                    Issue(
                        "error",
                        path,
                        line_number,
                        f"broken relative Markdown link: {target}",
                    )
                )
    return issues


def git_lines(root: Path, *args: str) -> list[str]:
    output = run_git(root, *args, check=False)
    return [line for line in output.splitlines() if line]


def selected_paths(root: Path, mode: str, base: str) -> list[str]:
    """Select tracked/untracked paths for staged, changed, base, or all mode."""

    if mode == "all":
        return git_lines(root, "ls-files")
    if mode == "staged":
        return git_lines(
            root, "diff", "--cached", "--name-only", "--diff-filter=ACMR"
        )
    if mode == "base":
        return git_lines(
            root, "diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"
        )
    changed = git_lines(root, "diff", "--name-only", "--diff-filter=ACMR", "HEAD")
    untracked = git_lines(root, "ls-files", "--others", "--exclude-standard")
    return sorted(set(changed + untracked))


def read_current(root: Path, path: str, mode: str) -> str | None:
    if mode == "staged":
        result = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None
    full = root / path
    if not full.is_file():
        return None
    return full.read_text(encoding="utf-8", errors="replace")


def read_base(root: Path, path: str, base_ref: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path}"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def staged_entries(root: Path) -> dict[str, tuple[str, int]]:
    """Return staged paths mapped to `(git mode, blob size)`.

    Staged checks must describe what will be committed, even if the working
    copy was deleted or edited after staging. Reading mode and size from the
    index prevents the working tree from masking staged content.
    """

    entries: dict[str, tuple[str, int]] = {}
    for line in git_lines(root, "ls-files", "--stage"):
        metadata, separator, path = line.partition("\t")
        if not separator:
            continue
        parts = metadata.split()
        if len(parts) != 3:
            continue
        if parts[0] == "160000":  # Gitlink: the index object is a commit, not a blob.
            entries[path] = (parts[0], 0)
            continue
        size_result = subprocess.run(
            ["git", "cat-file", "-s", f":{path}"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if size_result.returncode != 0:
            continue
        entries[path] = (parts[0], int(size_result.stdout.strip()))
    return entries


def staged_path_exists(root: Path, entries: dict[str, tuple[str, int]], candidate: Path) -> bool:
    """Return whether a file or directory exists in the Git index."""

    try:
        relative = candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.exists()
    if relative == ".":
        return True
    prefix = relative.rstrip("/") + "/"
    if relative in entries or any(path.startswith(prefix) for path in entries):
        return True
    # A submodule appears in the superproject index as one Gitlink entry, not
    # as entries for its contained files. Resolve paths below that ancestor
    # in the checked-out submodule working tree.
    parts = Path(relative).parts
    ancestors = ("/".join(parts[:index]) for index in range(1, len(parts)))
    if any(ancestor in entries for ancestor in ancestors):
        return candidate.exists()
    return False


def check_primary_types(
    *,
    path: str,
    current: str,
    base_text: str | None,
    inventory_only: bool,
) -> list[Issue]:
    """Apply the one-primary-type new-file and anti-growth rules."""

    if not is_production_source(path) or has_exception(current, "primary-type"):
        return []
    current_types = primary_types(path, current)
    base_types = [] if base_text is None else primary_types(path, base_text)
    if len(current_types) <= 1:
        return []
    joined = ", ".join(current_types)
    if inventory_only:
        return [
            Issue(
                "warning",
                path,
                None,
                f"legacy file has {len(current_types)} primary types: {joined}",
            )
        ]
    if base_text is None:
        return [
            Issue(
                "error",
                path,
                None,
                f"new production file declares {len(current_types)} primary types: {joined}",
            )
        ]
    if len(current_types) > len(base_types):
        return [
            Issue(
                "error",
                path,
                None,
                "primary type count grew from "
                f"{len(base_types)} to {len(current_types)}: {joined}",
            )
        ]
    return []


def check_public_docs(
    *,
    path: str,
    current: str,
    base_text: str | None,
    inventory_only: bool,
) -> list[Issue]:
    """Require docs for newly added public Swift/Python declarations."""

    if not is_production_source(path) or has_exception(current, "public-docs"):
        return []
    issues: list[Issue] = []
    severity = "warning" if inventory_only else "error"

    if path.endswith(".swift"):
        current_decls = swift_public_declarations(current)
        base_identities = {
            identity
            for identity, _, _ in swift_public_declarations(base_text or "")
        }
        lines = current.splitlines()
        for identity, line, name in current_decls:
            is_new = identity not in base_identities
            if (is_new or inventory_only) and not swift_declaration_is_documented(
                lines, line
            ):
                issues.append(
                    Issue(
                        severity,
                        path,
                        line,
                        f"public Swift declaration `{name}` needs a documentation comment",
                    )
                )
        return issues

    if path.endswith(".py") and "mlx_shap_e/" in path:
        current_decls = python_public_declarations(current)
        base_decls = python_public_declarations(base_text or "")
        for qualified, (line, documented) in current_decls.items():
            is_new = qualified not in base_decls
            if (is_new or inventory_only) and not documented:
                issues.append(
                    Issue(
                        severity,
                        path,
                        line,
                        f"public Python declaration `{qualified}` needs a docstring",
                    )
                )
    return issues


def companion_warning(paths: list[str]) -> Issue | None:
    """Warn when production changes arrive without tests or documentation."""

    production = [path for path in paths if is_production_source(path)]
    if not production:
        return None
    companion = [
        path
        for path in paths
        if path.endswith(".md")
        or "/Tests/" in path
        or path.startswith("tests/")
        or "/cli/check/" in path
        or "run_parity" in path
        or "run_image_parity" in path
    ]
    if companion:
        return None
    return Issue(
        "warning",
        production[0],
        None,
        "production source changed without a test or documentation change in "
        "the same change; record why coverage/docs already suffice",
    )


def run(root: Path, mode: str, base: str) -> list[Issue]:
    """Run all selected checks and return every finding."""

    paths = selected_paths(root, mode, base)
    base_ref = base if mode == "base" else "HEAD"
    inventory_only = mode == "all"
    staged = staged_entries(root) if mode == "staged" else {}
    issues: list[Issue] = []

    for path in paths:
        if mode == "staged":
            entry = staged.get(path)
            if entry is None or entry[0] == "120000":
                continue
            file_size = entry[1]
        else:
            full = root / path
            if not full.is_file() or full.is_symlink():
                continue
            file_size = full.stat().st_size
        if file_size > MAX_FILE_BYTES:
            issues.append(
                Issue(
                    "error" if not inventory_only else "warning",
                    path,
                    None,
                    f"file exceeds {MAX_FILE_BYTES} bytes; use an external artifact store",
                )
            )
            continue
        if Path(path).suffix.lower() not in TEXT_SUFFIXES and Path(path).name not in {
            "AGENTS.md",
            "CLAUDE.md",
        }:
            continue
        current = read_current(root, path, mode)
        if current is None:
            if mode == "staged":
                issues.append(
                    Issue(
                        "error",
                        path,
                        None,
                        "could not read staged content from the Git index",
                    )
                )
            continue
        base_text = None if inventory_only else read_base(root, path, base_ref)

        issues.extend(find_secret_issues(path, current))
        if path.endswith(".md"):
            exists = (
                (lambda candidate: staged_path_exists(root, staged, candidate))
                if mode == "staged"
                else None
            )
            issues.extend(
                find_markdown_link_issues(root, path, current, path_exists=exists)
            )
        issues.extend(
            check_primary_types(
                path=path,
                current=current,
                base_text=base_text,
                inventory_only=inventory_only,
            )
        )
        issues.extend(
            check_public_docs(
                path=path,
                current=current,
                base_text=base_text,
                inventory_only=inventory_only,
            )
        )

    if not inventory_only:
        warning = companion_warning(paths)
        if warning is not None:
            issues.append(warning)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--changed", dest="mode", action="store_const", const="changed")
    group.add_argument("--staged", dest="mode", action="store_const", const="staged")
    group.add_argument("--all", dest="mode", action="store_const", const="all")
    group.add_argument("--base", metavar="REF", help="Compare REF...HEAD")
    parser.set_defaults(mode="changed")
    args = parser.parse_args()

    root = Path(run_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    mode = "base" if args.base else args.mode
    try:
        issues = run(root, mode, args.base or "HEAD")
    except RuntimeError as exc:
        print(f"ERROR standards checker: {exc}", file=sys.stderr)
        return 2

    for issue in issues:
        print(issue.render())
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    print(f"Coding standards: {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
