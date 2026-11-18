import argparse
from pathlib import Path

import numpy as np
import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import (
    DEFAULT_KARRAS_S_CHURN,
    DEFAULT_KARRAS_SIGMA_MAX,
    DEFAULT_KARRAS_SIGMA_MIN,
    sample_latents,
)
from shap_e.models.download import load_config, load_model

from cli.run.decode_text300m_latent_to_mesh import decode_latent_mesh


def _slugify(text: str, max_len: int = 40) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return (slug or "prompt")[:max_len]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference Shap-E text300M -> latent -> mesh generation (PyTorch diffusion)."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument("--guidance-scale", type=float, default=15.0, help="Classifier-free guidance scale.")
    parser.add_argument("--karras-steps", type=int, default=64, help="Number of Karras sampling steps.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Path to shap_e cache directory (defaults to repo_root/shap_e_model_cache).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated artifacts (defaults to shap_e/outputs/reference).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else repo_root / "shap_e_model_cache"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root / "shap_e" / "outputs" / "reference"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(args.prompt)
    latent_path = output_dir / f"{slug}_latent.npz"
    ply_path = output_dir / f"{slug}.ply"
    obj_path = output_dir / f"{slug}.obj"
    mesh_npz_path = output_dir / f"{slug}_mesh.npz"

    device = torch.device("cpu")

    print("Loading text300M model...")
    model = load_model("text300M", device=device, cache_dir=str(cache_dir))

    print("Loading diffusion config...")
    diffusion_config = load_config("diffusion", cache_dir=str(cache_dir))
    diffusion = diffusion_from_config(diffusion_config)

    print("Sampling latent with reference diffusion...")
    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        model_kwargs=dict(texts=[args.prompt]),
        guidance_scale=args.guidance_scale,
        clip_denoised=True,
        use_fp16=False,
        use_karras=True,
        karras_steps=args.karras_steps,
        sigma_min=DEFAULT_KARRAS_SIGMA_MIN,
        sigma_max=DEFAULT_KARRAS_SIGMA_MAX,
        s_churn=DEFAULT_KARRAS_S_CHURN,
        device=device,
        progress=True,
    )

    latent = latents[0].detach().cpu()
    latent_np = latent.numpy().astype(np.float32)
    np.savez_compressed(
        latent_path,
        prompt=np.array([args.prompt]),
        seed=np.array([args.seed], dtype=np.int32),
        guidance_scale=np.array([args.guidance_scale], dtype=np.float32),
        karras_steps=np.array([args.karras_steps], dtype=np.int32),
        latent_flat=latent_np[None, :],
    )

    print("Loading decoder model...")
    decoder = load_model("decoder", device=device, cache_dir=str(cache_dir))

    print("Decoding mesh...")
    mesh = decode_latent_mesh(decoder, latent_np).tri_mesh()
    with ply_path.open("wb") as f:
        mesh.write_ply(f)
    with obj_path.open("w", encoding="utf-8") as f:
        mesh.write_obj(f)
    mesh.save(str(mesh_npz_path))

    print("Reference generation complete:")
    print(f"- latent: {latent_path}")
    print(f"- mesh ply: {ply_path}")
    print(f"- mesh obj: {obj_path}")
    print(f"- mesh npz: {mesh_npz_path}")
    print(f"verts={mesh.verts.shape[0]}, faces={mesh.faces.shape[0]}")


if __name__ == "__main__":
    main()
