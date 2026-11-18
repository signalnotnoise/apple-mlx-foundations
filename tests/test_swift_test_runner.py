"""Regression tests for the ShapERuntime shell test runner."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "shap_e" / "swift" / "run_tests.sh"


def _run_with_stub(tmp_path: Path, *, fail_first: bool) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    count_path = tmp_path / "count"
    xcodebuild = bin_dir / "xcodebuild"
    xcodebuild.write_text(
        "#!/bin/sh\n"
        f"count_file={str(count_path)!r}\n"
        "count=$(cat \"$count_file\" 2>/dev/null || echo 0)\n"
        "count=$((count + 1))\n"
        "printf '%s' \"$count\" > \"$count_file\"\n"
        "echo invocation:$count\n"
        + ("if [ \"$count\" -eq 1 ]; then exit 65; fi\n" if fail_first else "")
        + "exit 0\n",
        encoding="utf-8",
    )
    xcbeautify = bin_dir / "xcbeautify"
    xcbeautify.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
    xcodebuild.chmod(xcodebuild.stat().st_mode | stat.S_IXUSR)
    xcbeautify.chmod(xcbeautify.stat().st_mode | stat.S_IXUSR)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = subprocess.run(
        ["/bin/bash", str(RUNNER)],
        cwd=RUNNER.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    result_count = int(count_path.read_text(encoding="utf-8"))
    assert result_count == 1, result.stdout + result.stderr
    return result


def test_runner_does_not_retry_a_failed_test_run(tmp_path: Path) -> None:
    result = _run_with_stub(tmp_path, fail_first=True)
    assert result.returncode == 65
    assert result.stdout.count("invocation:") == 1


def test_runner_runs_successful_test_once(tmp_path: Path) -> None:
    result = _run_with_stub(tmp_path, fail_first=False)
    assert result.returncode == 0
    assert result.stdout.count("invocation:") == 1
