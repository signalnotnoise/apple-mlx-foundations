from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shap_e" / "src"))

from cli.run.download_shap_e_bundle import (  # noqa: E402
    IMAGE_FAMILIES,
    TEXT_FAMILIES,
    ensure_text_only_output_is_clean,
    parse_checksum_lines,
    patterns_for_scope,
    verify_checksums,
)


def test_full_download_uses_complete_snapshot() -> None:
    assert patterns_for_scope(text_only=False) is None


def test_text_only_patterns_exclude_image_families() -> None:
    patterns = patterns_for_scope(text_only=True)
    assert patterns is not None
    for family in TEXT_FAMILIES:
        assert f"{family}/*.bin" in patterns
    for family in IMAGE_FAMILIES:
        assert f"{family}/*.bin" not in patterns
    assert "manifest.json" in patterns
    assert "bpe_simple_vocab_16e6.txt.gz" in patterns


def test_text_only_output_rejects_existing_image_family(tmp_path: Path) -> None:
    family = tmp_path / "clip_vision"
    family.mkdir()
    (family / "tensor.bin").write_bytes(b"not empty")
    with pytest.raises(ValueError, match="already contains image families"):
        ensure_text_only_output_is_clean(tmp_path)


def test_parse_checksum_lines_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_checksum_lines(["not-a-checksum"])


def _write_bundle(root: Path, files: dict[str, bytes]) -> None:
    lines = []
    for name, payload in sorted(files.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_verify_checksums_accepts_complete_bundle(tmp_path: Path) -> None:
    _write_bundle(tmp_path, {"manifest.json": b"{}", "clip_text/a.bin": b"abc"})
    assert verify_checksums(tmp_path) == 2


def test_verify_checksums_rejects_corruption(tmp_path: Path) -> None:
    _write_bundle(tmp_path, {"clip_text/a.bin": b"abc"})
    (tmp_path / "clip_text/a.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checksums(tmp_path)


def test_verify_checksums_partial_skips_only_missing_files(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        {
            "clip_text/a.bin": b"abc",
            "clip_vision/b.bin": b"image-only",
        },
    )
    (tmp_path / "clip_vision/b.bin").unlink()
    assert verify_checksums(tmp_path, require_all=False) == 1
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_checksums(tmp_path, require_all=True)
