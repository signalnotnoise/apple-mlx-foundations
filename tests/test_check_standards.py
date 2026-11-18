from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_standards.py"
SPEC = importlib.util.spec_from_file_location("check_standards", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check
SPEC.loader.exec_module(check)


def test_primary_type_parsers() -> None:
    swift = """
public struct First {}
public final class Second {}
private struct NestedScopeExample {}
"""
    # NestedScopeExample is still top-level in this fixture; indentation is
    # what the parser uses to distinguish nested declarations in real files.
    assert check.swift_primary_types(swift) == [
        "First",
        "Second",
        "NestedScopeExample",
    ]
    python = "class First:\n    pass\n\nclass Second:\n    pass\n"
    assert check.python_primary_classes(python) == ["First", "Second"]


def test_new_file_primary_type_limit_and_exception() -> None:
    path = "shap_e/swift/Sources/ShapERuntime/Example.swift"
    current = "public struct One {}\npublic struct Two {}\n"
    issues = check.check_primary_types(
        path=path, current=current, base_text=None, inventory_only=False
    )
    assert len(issues) == 1 and issues[0].severity == "error"

    justified = "// standards-exception: primary-type cohesive result types\n" + current
    assert (
        check.check_primary_types(
            path=path, current=justified, base_text=None, inventory_only=False
        )
        == []
    )


def test_existing_file_primary_type_growth_is_rejected() -> None:
    path = "shap_e/src/mlx_shap_e/example.py"
    issues = check.check_primary_types(
        path=path,
        current="class One:\n    pass\n\nclass Two:\n    pass\n",
        base_text="class One:\n    pass\n",
        inventory_only=False,
    )
    assert len(issues) == 1 and "count grew" in issues[0].message


def test_new_public_swift_declaration_requires_docs() -> None:
    path = "shap_e/swift/Sources/ShapERuntime/Example.swift"
    missing = check.check_public_docs(
        path=path,
        current="public struct Example {}\n",
        base_text="",
        inventory_only=False,
    )
    assert len(missing) == 1

    documented = check.check_public_docs(
        path=path,
        current="/// Example value.\npublic struct Example {}\n",
        base_text="",
        inventory_only=False,
    )
    assert documented == []


def test_new_public_python_declaration_requires_docstring() -> None:
    path = "shap_e/src/mlx_shap_e/example.py"
    missing = check.check_public_docs(
        path=path,
        current="def run():\n    return 1\n",
        base_text="",
        inventory_only=False,
    )
    assert len(missing) == 1

    documented = check.check_public_docs(
        path=path,
        current='def run():\n    """Run it."""\n    return 1\n',
        base_text="",
        inventory_only=False,
    )
    assert documented == []


def test_secret_scan_allows_git_transport_and_markdown_history() -> None:
    home = "/" + "Users/example/project"
    email = "person@" + "old.example"
    source = f'path = "{home}"\ncontact = "{email}"\n'
    source_issues = check.find_secret_issues("example.py", source)
    assert len(source_issues) == 2

    old = "a@" + "b.co"
    markdown = f"Clone with git@github.com:owner/repo.git; old contact {old}\n"
    assert check.find_secret_issues("README.md", markdown) == []


def test_markdown_link_validation(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Target\n", encoding="utf-8")
    issues = check.find_markdown_link_issues(
        tmp_path,
        "README.md",
        "[ok](target.md) [bad](missing.md) [web](https://example.com)\n",
    )
    assert len(issues) == 1
    assert "missing.md" in issues[0].message


def test_staged_path_exists_under_submodule_gitlink(tmp_path: Path) -> None:
    target = tmp_path / "TripoSR" / "README.md"
    target.parent.mkdir()
    target.write_text("# TripoSR\n", encoding="utf-8")
    entries = {"TripoSR": ("160000", 0)}

    assert check.staged_path_exists(tmp_path, entries, target)
    assert not check.staged_path_exists(
        tmp_path, entries, tmp_path / "TripoSR" / "missing.md"
    )


def test_staged_mode_reads_index_after_worktree_removal(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, stdout=subprocess.DEVNULL
        )

    git("init", "-q")
    git(
        "-c",
        "user.name=Review",
        "-c",
        "user.email=review@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "baseline",
    )
    document = tmp_path / "example.md"
    document.write_text("[missing](missing.md)\n", encoding="utf-8")
    git("add", document.name)

    present_issues = check.run(tmp_path, "staged", "HEAD")
    assert any("broken relative Markdown link" in issue.message for issue in present_issues)

    document.unlink()
    missing_issues = check.run(tmp_path, "staged", "HEAD")
    assert any("broken relative Markdown link" in issue.message for issue in missing_issues)


def test_companion_warning_for_source_without_tests_or_docs() -> None:
    warning = check.companion_warning(
        ["shap_e/swift/Sources/ShapERuntime/ImageProcessor.swift"]
    )
    assert warning is not None and warning.severity == "warning"
    assert (
        check.companion_warning(
            [
                "shap_e/swift/Sources/ShapERuntime/ImageProcessor.swift",
                "shap_e/swift/Tests/ShapERuntimeTests/ImageProcessorTests.swift",
            ]
        )
        is None
    )
