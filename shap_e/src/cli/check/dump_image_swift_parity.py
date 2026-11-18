"""
Dump upstream Torch image-conditioning tensors for Swift parity.

The production Python MLX runtime is text-only, so this script deliberately
uses the upstream OpenAI CLIP and Shap-E image300M implementations as the
independent oracle for the Swift-only image path.

Usage (from shap_e/src):
  python -m cli.check.dump_image_swift_parity

Docs: shap_e/parity/README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cli.check.color import GREEN, colored_print


def synthetic_image(seed: int = 7, width: int = 448, height: int = 336) -> Image.Image:
    """Create a deterministic, non-square RGB test image.

    The shape intentionally exercises CLIP Resize(224) + CenterCrop(224), and
    the high-contrast geometry makes accidental axis, channel, or crop errors
    visible in the processed-tensor comparison.
    """

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    image[..., 1] = (yy * 255 // max(height - 1, 1)).astype(np.uint8)
    image[..., 2] = ((np.sin(xx / 23.0) + np.cos(yy / 19.0)) * 63 + 128).astype(np.uint8)
    image[::16, :, :] = 255 - image[::16, :, :]
    image[:, ::16, :] = 255 - image[:, ::16, :]

    pil = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(pil)
    draw.ellipse((width * 0.18, height * 0.16, width * 0.72, height * 0.82), fill=(235, 64, 40))
    draw.rectangle((width * 0.58, height * 0.48, width * 0.94, height * 0.92), fill=(30, 90, 230))
    draw.line((0, height - 1, width - 1, 0), fill=(255, 230, 40), width=9)
    # A small deterministic texture patch helps expose accidental smoothing or
    # a flipped axis without making the whole comparison noise-dominated.
    patch = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    pil.paste(Image.fromarray(patch, mode="RGB"), (width - 72, 24))
    return pil


def upstream_clip_grid(clip_model, clip_input):
    """Run OpenAI CLIP's visual grid path, returning NLC float32 output.

    This follows `ImageCLIP.embed_images_grid` in upstream Shap-E: patch
    convolution, class token, position embedding, pre-LN, transformer, then
    drop the class token. It intentionally does not apply `ln_post` or the
    pooled-output projection.
    """

    import torch

    visual = clip_model.visual
    x = clip_input.to(clip_model.dtype)
    x = visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    class_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    return x[:, 1:, :].contiguous().float()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump Torch CLIP-grid and image300M tensors for Swift image parity."
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: shap_e/parity).")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Shap-E/CLIP checkpoint cache (default: repo shap_e_model_cache).",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional input image. Default is a deterministic synthetic RGB image.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for synthetic image and latent.")
    parser.add_argument(
        "--timestep", type=float, default=500.0, help="Diffusion timestep for the forward check."
    )
    args = parser.parse_args()

    import clip
    import torch
    from shap_e.models.download import load_model

    repo_root = Path(__file__).resolve().parents[4]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else repo_root / "shap_e" / "parity"
    )
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else repo_root / "shap_e_model_cache"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        image = Image.open(Path(args.image_path).expanduser().resolve()).convert("RGB")
        image_description = str(Path(args.image_path).expanduser().resolve())
    else:
        image = synthetic_image(seed=args.seed)
        image_description = f"synthetic seed={args.seed} size={image.size}"

    colored_print(f"Image case: {image_description}", GREEN)
    colored_print(f"Cache:      {cache_dir}", GREEN)
    colored_print(f"Out:        {out_dir}", GREEN)

    device = torch.device("cpu")
    clip_model, preprocess = clip.load(
        "ViT-L/14", device=device, download_root=str(cache_dir)
    )
    clip_model.eval()

    # Save the exact shared input in both the raw RGB form used to make a
    # CGImage in Swift and the NHWC preprocessed form consumed by Swift's
    # ClipImageEmbedder. The first isolates preprocessing; the second isolates
    # the vision transformer from platform image-resampling differences.
    raw_rgb = np.asarray(image).astype(np.float32)
    np.save(out_dir / "python_image_rgb.npy", raw_rgb)
    clip_input_chw = preprocess(image).unsqueeze(0).float()
    clip_input_nhwc = clip_input_chw.permute(0, 2, 3, 1).contiguous().numpy()
    np.save(out_dir / "python_clip_input.npy", clip_input_nhwc)

    with torch.no_grad():
        grid_nlc = upstream_clip_grid(clip_model, clip_input_chw)
    grid = grid_nlc.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "python_image_grid.npy", grid)
    colored_print(f"  clip_input NHWC {clip_input_nhwc.shape}", GREEN)
    colored_print(f"  image_grid NLC  {grid.shape}", GREEN)

    colored_print("Loading upstream image300M (Torch)...", GREEN)
    diffusion_model = load_model("image300M", device=device, cache_dir=str(cache_dir))
    diffusion_model.eval()
    inner = diffusion_model.wrapped

    rng = np.random.default_rng(args.seed)
    latent_nct = rng.standard_normal((1, 1024, 1024), dtype=np.float32)
    timestep = np.asarray([args.timestep], dtype=np.float32)
    np.save(out_dir / "python_image_latent.npy", latent_nct)
    np.save(out_dir / "python_image_timestep.npy", timestep)

    with torch.no_grad():
        output = inner(
            torch.from_numpy(latent_nct),
            torch.from_numpy(timestep),
            embeddings=grid_nlc.permute(0, 2, 1).contiguous(),
        )
    output_np = output.detach().cpu().numpy().astype(np.float32)
    np.save(out_dir / "python_image300m_output.npy", output_np)
    colored_print(f"  image300m input  {latent_nct.shape}", GREEN)
    colored_print(f"  image300m output {output_np.shape}", GREEN)

    summary = {
        "image": image_description,
        "image_size_hw": [int(image.height), int(image.width)],
        "seed": int(args.seed),
        "timestep": float(args.timestep),
        "clip_input_shape_nhwc": list(clip_input_nhwc.shape),
        "image_grid_shape_nlc": list(grid.shape),
        "image_latent_shape_nct": list(latent_nct.shape),
        "image300m_output_shape_nct": list(output_np.shape),
    }
    (out_dir / "python_image_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    colored_print(f"Wrote Torch image parity dumps to {out_dir}", GREEN)


if __name__ == "__main__":
    main()
