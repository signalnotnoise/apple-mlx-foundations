import argparse
from collections import Counter
from pathlib import Path
import sys

import torch
from requests.exceptions import RequestException
from shap_e.models.download import load_model


def prefixes(key: str, max_depth: int = 4):
    parts = key.split(".")
    for depth in range(1, min(max_depth, len(parts)) + 1):
        yield ".".join(parts[:depth])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a Shap-E checkpoint and emit key/shape reports."
    )
    parser.add_argument(
        "--model",
        default="text300M",
        choices=["text300M", "transmitter", "diffusion"],
        help="Shap-E checkpoint name to inspect.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated reports. Defaults to the shap_e module root.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local cache directory used by shap_e download helpers.",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    default_output_dir = src_dir.parent
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PyTorch checkpoint: {args.model}")
    try:
        pt_model = load_model(
            args.model,
            device=torch.device("cpu"),
            cache_dir=args.cache_dir,
        )
    except RequestException as exc:
        print(f"Failed to fetch checkpoint/config for '{args.model}': {exc}")
        print(
            "Tip: retry with network access or use --cache-dir pointing to a "
            "directory that already contains the Shap-E artifacts."
        )
        sys.exit(1)

    state_dict = pt_model.state_dict()

    key_file = output_dir / f"{args.model}_keys.txt"
    shape_file = output_dir / f"{args.model}_tensor_shapes.txt"
    prefix_file = output_dir / f"{args.model}_prefix_counts.txt"

    keys = sorted(state_dict.keys())

    with key_file.open("w", encoding="utf-8") as f:
        for key in keys:
            f.write(f"{key}\n")

    with shape_file.open("w", encoding="utf-8") as f:
        for key in keys:
            shape = tuple(state_dict[key].shape)
            f.write(f"{key}\t{shape}\n")

    counter: Counter[str] = Counter()
    for key in keys:
        counter.update(prefixes(key, max_depth=4))

    with prefix_file.open("w", encoding="utf-8") as f:
        for prefix, count in sorted(counter.items()):
            f.write(f"{prefix}\t{count}\n")

    print("Analysis complete.")
    print(f"- keys: {key_file}")
    print(f"- tensor shapes: {shape_file}")
    print(f"- prefix counts: {prefix_file}")


if __name__ == "__main__":
    main()
