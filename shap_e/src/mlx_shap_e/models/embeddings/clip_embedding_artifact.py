from __future__ import annotations

from pathlib import Path

import numpy as np


def load_clip_embedding_artifact(
    artifact_path: str | Path,
    *,
    key: str = "clip_embedding",
) -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[2]
    path = Path(artifact_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CLIP embedding artifact not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        emb = np.load(path)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                available = ", ".join(sorted(data.files))
                raise KeyError(
                    f"Embedding key '{key}' not found in {path}. "
                    f"Available keys: {available}"
                )
            emb = data[key]
    else:
        raise ValueError(f"Unsupported embedding artifact format: {path}")

    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb[None, :]
    if emb.ndim != 2 or emb.shape[-1] != 768:
        raise ValueError(
            f"Expected CLIP embedding shape [batch, 768] or [768], got {tuple(emb.shape)} from {path}"
        )
    if emb.shape[0] != 1:
        raise ValueError(
            f"Expected single-prompt embedding with batch=1, got shape {tuple(emb.shape)} from {path}"
        )
    return emb
