from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_triposr_weights.py"

spec = importlib.util.spec_from_file_location("package_triposr_weights", SCRIPT)
assert spec is not None and spec.loader is not None
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


def write_fake_safetensors(path: Path, tensor_names: list[str]) -> None:
    """Write a minimal safetensors file with a real header and zero data."""
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        for name in tensor_names
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 4)


def test_safetensors_tensor_count_ignores_metadata(tmp_path: Path) -> None:
    weights = tmp_path / "w.safetensors"
    write_fake_safetensors(weights, ["a.weight", "b.weight"])
    assert packaging.safetensors_tensor_count(weights) == 2


def test_write_checksums_covers_all_but_itself(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta")
    packaging.write_checksums(tmp_path)
    lines = (tmp_path / "checksums.sha256").read_text().splitlines()
    recorded = {line.split("  ")[1]: line.split("  ")[0] for line in lines}
    assert recorded == {
        "a.txt": hashlib.sha256(b"alpha").hexdigest(),
        "sub/b.txt": hashlib.sha256(b"beta").hexdigest(),
    }


@pytest.mark.skipif(
    not (ROOT / "TripoSR" / "LICENSE").exists(),
    reason="TripoSR submodule not initialized",
)
def test_staging_builds_bundle_and_refuses_existing_dir(tmp_path: Path) -> None:
    weights = tmp_path / "triposr_swift_ready.safetensors"
    intermediate = tmp_path / "triposr_mlx.safetensors"
    ckpt = tmp_path / "model.ckpt"
    validation = tmp_path / "parity-summary.json"
    write_fake_safetensors(weights, ["only.weight"])
    write_fake_safetensors(intermediate, ["only.weight", "extra.weight"])
    ckpt.write_bytes(b"fake-checkpoint")
    validation.write_text("{}\n")

    staging = tmp_path / "staging"
    command = [
        sys.executable,
        str(SCRIPT),
        "--staging",
        str(staging),
        "--weights",
        str(weights),
        "--intermediate",
        str(intermediate),
        "--ckpt",
        str(ckpt),
        "--validation",
        str(validation),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    provenance = json.loads((staging / "provenance.json").read_text())
    assert provenance["canonical_artifact"]["tensors"] == 1
    assert provenance["intermediate_not_published"]["tensors"] == 2
    assert provenance["upstream"]["revision"] == packaging.UPSTREAM_REVISION
    assert (staging / "licenses" / "TripoSR.txt").exists()
    assert (staging / "validation" / "parity-summary.json").exists()

    checksums = (staging / "checksums.sha256").read_text()
    assert "triposr_swift_ready.safetensors" in checksums
    assert "checksums.sha256" not in checksums

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stderr
