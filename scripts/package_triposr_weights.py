#!/usr/bin/env python3
"""Stage the TripoSR MLX weights bundle for a Hugging Face upload.

The canonical published artifact is ``triposr_swift_ready.safetensors`` —
the only weight file the Swift runtime loads. The intermediate
``triposr_mlx.safetensors`` pack is recorded in provenance but never
published. This script copies the canonical file into a fresh staging
directory and writes the release metadata around it: ``provenance.json``,
``checksums.sha256``, licenses/notice, the model card, a re-export driver,
and any ``validation/`` summaries the caller supplies.

The staging directory must not already exist, so a stale bundle can never
be uploaded by accident. Upload and fresh-download verification are
deliberately separate steps; see ``docs/RELEASE-STATUS.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPO = "stabilityai/TripoSR"
UPSTREAM_REVISION = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
CANONICAL_NAME = "triposr_swift_ready.safetensors"
INTERMEDIATE_NAME = "triposr_mlx.safetensors"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path`` read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_tensor_count(path: Path) -> int:
    """Return the tensor count from a safetensors header (no torch needed)."""
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return sum(1 for key in header if key != "__metadata__")


def git_commit(repo: Path) -> str:
    """Return the HEAD commit of the git checkout at ``repo``."""
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def file_record(path: Path) -> dict:
    """Return name/bytes/sha256 (and tensor count for safetensors)."""
    record = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".safetensors":
        record["tensors"] = safetensors_tensor_count(path)
    return record


def resolve_upstream_ckpt(explicit: Path | None) -> Path:
    """Locate the pinned upstream ``model.ckpt`` for provenance hashing."""
    if explicit is not None:
        return explicit
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--TripoSR/snapshots"
        / UPSTREAM_REVISION
        / "model.ckpt"
    )
    if snapshot.exists():
        return snapshot
    raise SystemExit(
        "Upstream model.ckpt not found in the Hugging Face cache; "
        "pass --ckpt or run TripoSR/export_tripo_weights.py first."
    )


MODEL_CARD = """---
license: mit
base_model: stabilityai/TripoSR
pipeline_tag: image-to-3d
library_name: mlx
tags:
  - mlx
  - swift
  - image-to-3d
  - triposr
---

# TripoSR MLX weights (Swift)

Converted [TripoSR](https://huggingface.co/stabilityai/TripoSR) weights for
the MLX Swift runtime in
[apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations).
This is a format/layout conversion of the upstream checkpoint — no training
or fine-tuning. Conversion pins upstream revision `{upstream_revision}`.

## Files

| File | Role |
|---|---|
| `triposr_swift_ready.safetensors` | **Canonical artifact.** The only file the Swift runtime loads (keys/layouts remapped for the Swift modules; the loader verifies shapes and rejects unused keys). |
| `provenance.json` | Upstream checkpoint, intermediate pack, and derived file hashes; code commits; environment. |
| `checksums.sha256` | SHA-256 for every bundle file except itself. Verify before use. |
| `licenses/TripoSR.txt`, `LICENSE`, `NOTICE` | Upstream and bundle license texts and attribution. |
| `validation/` | Parity and mesh-smoke summaries for this bundle. |
| `reproduce_export.py` | How to regenerate the canonical file from the upstream checkpoint. |

The intermediate `triposr_mlx.safetensors` pack (upstream keys, pre-remap)
exists only between the two conversion scripts; it is recorded in
`provenance.json` and is **not** published.

## Usage (Swift)

```swift
import RuntimeWrapper

let runtime = try ThreeDGenerationRuntime(
    configuration: .tripoSR(weightsURL: tripoSRWeightsURL)
)
let meshes = try await runtime.generate(image: cgImage)
```

Or load the query-only runtime directly via `TripoSRModel.loadWeights(from:)`
in the `TripoSR` Swift package.

## Validation scope

- Python↔Swift tensor parity (tokens, backbone, triplanes, density, color):
  high correlation, not bit-exact — see `validation/parity-summary.json`.
- End-to-end Swift mesh smoke with these weights: see
  `validation/mesh-smoke.json`.
- Swift texture baking and background removal are not included; the Python
  `run.py` path covers those.

## License

MIT (see `LICENSE`). The upstream TripoSR release — source code and
pretrained models — is MIT, Copyright (c) 2024 Tripo AI & Stability AI;
see `licenses/TripoSR.txt` and `NOTICE`.
"""

NOTICE = """TripoSR MLX weights bundle
Copyright (c) 2026 signalnotnoise
Licensed under the MIT License; see LICENSE.

This bundle is a format/layout conversion of the TripoSR checkpoint. It is
not affiliated with or endorsed by Tripo AI, Stability AI, Meta AI, or Apple.

Third-party models this bundle derives from:

1. TripoSR (Tripo AI & Stability AI / VAST-AI-Research)
   https://github.com/VAST-AI-Research/TripoSR
   https://huggingface.co/stabilityai/TripoSR
   MIT License, Copyright (c) 2024 Tripo AI & Stability AI. The upstream
   release states the MIT license includes the source code and the
   pretrained models. License text: licenses/TripoSR.txt.

2. DINO ViT-B/16 image tokenizer (Meta AI)
   https://huggingface.co/facebook/dino-vitb16
   The TripoSR configuration initializes its image tokenizer from
   facebook/dino-vitb16; those parameters ship inside the upstream TripoSR
   checkpoint converted here, under the upstream release above.
"""

REPRODUCE = '''#!/usr/bin/env python3
"""Regenerate triposr_swift_ready.safetensors from the upstream checkpoint.

Clone https://github.com/signalnotnoise/TripoSR at the commit recorded in
provenance.json (field code.triposr_submodule), then run this file with the
checkout path. It invokes the two pinned conversion scripts; compare the
result against checksums.sha256.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    checkout = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("TripoSR")
    for script in ("export_tripo_weights.py", "convert_weights_to_mlx.py"):
        subprocess.run([sys.executable, script], cwd=checkout, check=True)
    print(checkout / "triposr_weights" / "triposr_swift_ready.safetensors")


if __name__ == "__main__":
    main()
'''


def write_checksums(staging: Path) -> None:
    """Write ``checksums.sha256`` covering every staged file but itself."""
    lines = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative == "checksums.sha256":
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    (staging / "checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Stage the bundle described in the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True,
                        help="New staging directory (must not exist)")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "TripoSR/triposr_weights" / CANONICAL_NAME)
    parser.add_argument("--intermediate", type=Path,
                        default=ROOT / "TripoSR/triposr_weights" / INTERMEDIATE_NAME)
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="Upstream model.ckpt (default: pinned HF cache snapshot)")
    parser.add_argument("--validation", type=Path, action="append", default=[],
                        help="Validation summary file to copy into validation/ (repeatable)")
    args = parser.parse_args()

    staging = args.staging
    if staging.exists():
        raise SystemExit(f"Staging directory already exists: {staging}")
    for required in (args.weights, args.intermediate):
        if not required.exists():
            raise SystemExit(f"Missing weight file: {required}")
    ckpt = resolve_upstream_ckpt(args.ckpt)

    (staging / "licenses").mkdir(parents=True)
    shutil.copy2(args.weights, staging / CANONICAL_NAME)
    shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
    shutil.copy2(ROOT / "TripoSR/LICENSE", staging / "licenses/TripoSR.txt")
    (staging / "NOTICE").write_text(NOTICE)
    (staging / "README.md").write_text(
        MODEL_CARD.format(upstream_revision=UPSTREAM_REVISION))
    (staging / "reproduce_export.py").write_text(REPRODUCE)

    if args.validation:
        (staging / "validation").mkdir()
        for path in args.validation:
            shutil.copy2(path, staging / "validation" / path.name)

    provenance = {
        "bundle": "triposr-mlx",
        "artifact_version": 1,
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_artifact": file_record(args.weights),
        "intermediate_not_published": file_record(args.intermediate),
        "upstream": {
            "repo": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "checkpoint": file_record(ckpt),
            "license": "MIT (source code and pretrained models, per upstream release)",
        },
        "conversion": {
            "kind": "format/layout export; not training or fine-tuning",
            "scripts": ["TripoSR/export_tripo_weights.py",
                        "TripoSR/convert_weights_to_mlx.py"],
            "packaging_script": "scripts/package_triposr_weights.py",
        },
        "code": {
            "apple_mlx_foundations": git_commit(ROOT),
            "triposr_submodule": git_commit(ROOT / "TripoSR"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    write_checksums(staging)
    staged = sorted(p.relative_to(staging).as_posix()
                      for p in staging.rglob("*") if p.is_file())
    print(f"Staged {len(staged)} files in {staging}:")
    for name in staged:
        print(f"  {name}")
    print(f"Canonical artifact sha256: {provenance['canonical_artifact']['sha256']}")


if __name__ == "__main__":
    main()
