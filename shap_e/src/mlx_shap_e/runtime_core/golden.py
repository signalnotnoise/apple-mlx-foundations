from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mlx_shap_e.models.embeddings.clip_tokenizer import ClipSimpleTokenizer, default_clip_bpe_path


@dataclass(frozen=True)
class RuntimeGoldenCase:
    prompt: str
    seed: int
    guidance_scale: float
    steps: int
    sigma_min: float
    sigma_max: float
    mesh_grid_size: int
    query_batch_size: int


def runtime_summary(
    *,
    case: RuntimeGoldenCase,
    token_ids: np.ndarray,
    embedding: np.ndarray,
    latent: np.ndarray,
    pred_xstart: np.ndarray,
    sigmas: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    artifact_version: int,
    artifact_family_versions: Dict[str, int],
) -> Dict[str, Any]:
    return {
        "case": asdict(case),
        "artifact_version": int(artifact_version),
        "artifact_family_versions": dict(artifact_family_versions),
        "token_ids": token_ids.astype(np.int32).tolist(),
        "clip_embedding_shape": list(embedding.shape),
        "clip_embedding_head": embedding.reshape(-1)[:8].astype(np.float32).tolist(),
        "latent_shape": list(latent.shape),
        "latent_head": latent.reshape(-1)[:8].astype(np.float32).tolist(),
        "pred_xstart_shape": list(pred_xstart.shape),
        "pred_xstart_head": pred_xstart.reshape(-1)[:8].astype(np.float32).tolist(),
        "sigmas": sigmas.astype(np.float32).tolist(),
        "mesh_vert_count": int(mesh_verts.shape[0]),
        "mesh_face_count": int(mesh_faces.shape[0]),
        "mesh_bbox_min": np.min(mesh_verts, axis=0).astype(np.float32).tolist() if mesh_verts.size else None,
        "mesh_bbox_max": np.max(mesh_verts, axis=0).astype(np.float32).tolist() if mesh_verts.size else None,
        "mesh_verts_head": mesh_verts[:4].astype(np.float32).tolist(),
        "mesh_faces_head": mesh_faces[:4].astype(np.int32).tolist(),
    }


def tokenize_prompt(prompt: str, *, context_length: int = 77, bpe_path: str | Path | None = None) -> np.ndarray:
    path = Path(bpe_path).expanduser().resolve() if bpe_path else default_clip_bpe_path()
    tokenizer = ClipSimpleTokenizer(path)
    return tokenizer.tokenize([prompt], context_length=context_length)


def load_golden(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def save_golden(path: str | Path, payload: Dict[str, Any]) -> None:
    Path(path).expanduser().resolve().write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compare_float_lists(
    actual: list[float],
    expected: list[float],
    *,
    atol: float = 1e-5,
) -> None:
    if len(actual) != len(expected):
        raise AssertionError(f"Length mismatch: {len(actual)} != {len(expected)}")
    for idx, (lhs, rhs) in enumerate(zip(actual, expected)):
        if abs(float(lhs) - float(rhs)) > atol:
            raise AssertionError(f"Float mismatch at index {idx}: actual={lhs} expected={rhs} atol={atol}")


def compare_runtime_summary(actual: Dict[str, Any], expected: Dict[str, Any], *, atol: float = 1e-5) -> None:
    exact_keys = [
        "case",
        "artifact_version",
        "token_ids",
        "clip_embedding_shape",
        "latent_shape",
        "pred_xstart_shape",
        "mesh_vert_count",
        "mesh_face_count",
    ]
    for key in exact_keys:
        if actual[key] != expected[key]:
            raise AssertionError(f"Mismatch for {key}: actual={actual[key]!r} expected={expected[key]!r}")

    # Family map may grow when new artifact families are added to the bundle
    # (e.g. clip_vision / image300m). Require expected keys/versions; allow extras.
    actual_families = dict(actual.get("artifact_family_versions") or {})
    expected_families = dict(expected.get("artifact_family_versions") or {})
    for fam, ver in expected_families.items():
        if fam not in actual_families:
            raise AssertionError(
                f"Missing artifact family {fam!r}: actual={actual_families!r} expected={expected_families!r}"
            )
        if actual_families[fam] != ver:
            raise AssertionError(
                f"Mismatch for artifact_family_versions[{fam!r}]: "
                f"actual={actual_families[fam]!r} expected={ver!r}"
            )

    float_keys = [
        "clip_embedding_head",
        "latent_head",
        "pred_xstart_head",
        "sigmas",
        "mesh_bbox_min",
        "mesh_bbox_max",
    ]
    for key in float_keys:
        compare_float_lists(actual[key], expected[key], atol=atol)

    nested_array_keys = [
        "mesh_verts_head",
        "mesh_faces_head",
    ]
    for key in nested_array_keys:
        actual_arr = np.asarray(actual[key])
        expected_arr = np.asarray(expected[key])
        if actual_arr.shape != expected_arr.shape:
            raise AssertionError(f"Shape mismatch for {key}: actual={actual_arr.shape} expected={expected_arr.shape}")
        if np.issubdtype(actual_arr.dtype, np.floating) or np.issubdtype(expected_arr.dtype, np.floating):
            if not np.allclose(actual_arr, expected_arr, atol=atol, rtol=0.0):
                raise AssertionError(f"Value mismatch for {key}: actual={actual[key]!r} expected={expected[key]!r}")
        elif not np.array_equal(actual_arr, expected_arr):
            raise AssertionError(f"Value mismatch for {key}: actual={actual[key]!r} expected={expected[key]!r}")
