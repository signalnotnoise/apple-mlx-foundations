from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


RUNTIME_ARTIFACT_VERSION = 1
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class TensorEntry:
    name: str
    path: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class ArtifactFamilyManifest:
    version: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: Dict[str, TensorEntry] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeArtifactManifest:
    artifact_version: int
    families: Dict[str, ArtifactFamilyManifest]


def manifest_path(bundle_dir: str | Path) -> Path:
    return Path(bundle_dir).expanduser().resolve() / _MANIFEST_NAME


def load_manifest(bundle_dir: str | Path) -> RuntimeArtifactManifest:
    path = manifest_path(bundle_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Runtime artifact manifest not found: {path}. "
            "Prepare artifacts offline before constructing the production runtime."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    families: Dict[str, ArtifactFamilyManifest] = {}
    for family_name, family_payload in payload.get("families", {}).items():
        tensors = {
            tensor_name: TensorEntry(
                name=tensor_name,
                path=str(tensor_payload["path"]),
                shape=[int(v) for v in tensor_payload["shape"]],
                dtype=str(tensor_payload["dtype"]),
            )
            for tensor_name, tensor_payload in family_payload.get("tensors", {}).items()
        }
        families[family_name] = ArtifactFamilyManifest(
            version=int(family_payload["version"]),
            metadata=dict(family_payload.get("metadata", {})),
            tensors=tensors,
        )
    return RuntimeArtifactManifest(
        artifact_version=int(payload.get("artifact_version", RUNTIME_ARTIFACT_VERSION)),
        families=families,
    )


def write_manifest(bundle_dir: str | Path, manifest: RuntimeArtifactManifest) -> Path:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    bundle_path.mkdir(parents=True, exist_ok=True)
    path = manifest_path(bundle_path)
    path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _safe_tensor_path(family: str, tensor_name: str) -> str:
    safe_name = tensor_name.replace("/", "__").replace(".", "__")
    return str(Path(family) / f"{safe_name}.bin")


def save_tensor_family(
    bundle_dir: str | Path,
    *,
    family: str,
    metadata: Dict[str, Any],
    tensors: Dict[str, np.ndarray],
    family_version: int = 1,
) -> Path:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    bundle_path.mkdir(parents=True, exist_ok=True)
    current = load_manifest(bundle_path) if manifest_path(bundle_path).exists() else RuntimeArtifactManifest(
        artifact_version=RUNTIME_ARTIFACT_VERSION,
        families={},
    )

    tensor_entries: Dict[str, TensorEntry] = {}
    for tensor_name, tensor_value in tensors.items():
        array = np.asarray(tensor_value)
        rel_path = _safe_tensor_path(family, tensor_name)
        tensor_path = bundle_path / rel_path
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        with tensor_path.open("wb") as handle:
            handle.write(array.tobytes(order="C"))
        tensor_entries[tensor_name] = TensorEntry(
            name=tensor_name,
            path=rel_path,
            shape=[int(v) for v in array.shape],
            dtype=str(array.dtype),
        )

    families = dict(current.families)
    families[family] = ArtifactFamilyManifest(
        version=int(family_version),
        metadata=dict(metadata),
        tensors=tensor_entries,
    )
    manifest = RuntimeArtifactManifest(
        artifact_version=current.artifact_version,
        families=families,
    )
    return write_manifest(bundle_path, manifest)
