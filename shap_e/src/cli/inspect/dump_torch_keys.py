import argparse
from pathlib import Path
import sys

import torch
from requests.exceptions import RequestException
from shap_e.models.download import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump checkpoint keys for a Shap-E model.")
    parser.add_argument(
        "--model",
        default="transmitter",
        choices=["text300M", "transmitter", "diffusion"],
        help="Checkpoint name passed to shap_e.models.download.load_model.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to '<model>_keys.txt' under shap_e/.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local cache directory used by shap_e download helpers.",
    )
    args = parser.parse_args()

    print(f"Loading PT model: {args.model}")
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

    src_dir = Path(__file__).resolve().parent
    default_output = src_dir.parent / f"{args.model}_keys.txt"
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for key in sorted(pt_model.state_dict().keys()):
            f.write(f"{key}\n")

    print(f"Done. Saved keys to {output_path}")


if __name__ == "__main__":
    main()
