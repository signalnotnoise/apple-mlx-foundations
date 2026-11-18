from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten


_ARTIFACT_TYPE = "mlx_text300m_v1"


def _to_path(key_str: str) -> list[object]:
    parts = key_str.split(".")
    path: list[object] = []
    for part in parts:
        path.append(int(part) if part.isdigit() else part)
    return path


def _to_key(path: Iterable[object]) -> str:
    return ".".join(str(part) for part in path)


@dataclass
class Text300MArtifact:
    n_ctx: int
    d_latent: int
    state: dict[str, np.ndarray]

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        *,
        regenerate_fn: Callable[[], None] | None = None,
    ) -> "Text300MArtifact":
        path = Path(artifact_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"MLX text300M artifact not found: {path}. "
                "Generate with: python -m cli.export.export_mlx_text300m_artifact"
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                artifact_type = str(data["artifact_type"].item())
                if artifact_type != _ARTIFACT_TYPE:
                    raise ValueError(f"Unsupported text300M artifact type: {artifact_type}")
                state = {k: np.asarray(v) for k, v in data.items() if k not in {"artifact_type"}}
        except zipfile.BadZipFile:
            if regenerate_fn is None:
                raise
            regenerate_fn()
            with np.load(path, allow_pickle=False) as data:
                artifact_type = str(data["artifact_type"].item())
                if artifact_type != _ARTIFACT_TYPE:
                    raise ValueError(f"Unsupported text300M artifact type: {artifact_type}")
                state = {k: np.asarray(v) for k, v in data.items() if k not in {"artifact_type"}}
        return cls(
            n_ctx=int(state.pop("meta_n_ctx").item()),
            d_latent=int(state.pop("meta_d_latent").item()),
            state=state,
        )

    def to_mlx_tree(self) -> dict:
        flat = [(k, mx.array(v.astype(np.float32))) for k, v in self.state.items()]
        return tree_unflatten(flat)


def save_text300m_artifact(
    artifact_path: Path,
    *,
    mlx_tree: dict,
    n_ctx: int,
    d_latent: int,
) -> None:
    payload: dict[str, np.ndarray] = {
        "artifact_type": np.array([_ARTIFACT_TYPE]),
        "meta_n_ctx": np.array([int(n_ctx)], dtype=np.int32),
        "meta_d_latent": np.array([int(d_latent)], dtype=np.int32),
    }
    for path, value in _flatten_tree(mlx_tree):
        payload[_to_key(path)] = np.asarray(value, dtype=np.float32)
    np.savez_compressed(artifact_path, **payload)


def _flatten_tree(tree: dict) -> list[tuple[list[object], np.ndarray]]:
    flat: list[tuple[list[object], np.ndarray]] = []

    def _walk(node: object, prefix: list[object]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, prefix + [key])
            return
        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                _walk(value, prefix + [idx])
            return
        flat.append((prefix, np.asarray(node)))

    _walk(tree, [])
    return flat
