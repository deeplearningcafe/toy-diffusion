import argparse
import os
import random
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from toy_diffusion.data.image import ImageDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    parser = argparse.ArgumentParser(
        description="Encode images into latents using a VAE."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing images."
    )
    parser.add_argument(
        "--vae_pretrained",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Pretrained VAE model name or path.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for encoding."
    )
    parser.add_argument(
        "--resize_dim", type=int, default=None, help="Resize images before encoding."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for VAE (float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--show_sample",
        action="store_true",
        help="Generate a grid plot with encoded and decoded images.",
    )
    parser.add_argument(
        "--show_encoded",
        action="store_true",
        help="Generate a grid plot with decoded latents.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Store the latents in float16 precision.",
    )
    parser.add_argument(
        "--load_into_ram",
        action="store_true",
        help="Store the images in ram",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)
    # Setup Mixed Precision
    capability = torch.cuda.get_device_capability() if device == "cuda" else (0, 0)
    autocast_dtype = torch.float32
    print(f"Autocast {autocast_dtype} and dtype {dtype}")

    if device == "cuda":
        if capability[0] >= 8:
            autocast_dtype = torch.bfloat16
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
        elif capability[0] == 7:
            autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")

    print(f"Loading VAE from {args.vae_pretrained}...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_pretrained, torch_dtype=dtype, cache_dir="models"
    ).to(device)
    vae.eval()
    vae = torch.compile(vae)

    vae_scale = 1.0  # getattr(vae.config, "scaling_factor", 1.0)
    vae_shift = 0.0  # getattr(vae.config, "shift_factor", 0.0)
    if vae_shift is None:
        vae_shift = 0.0
    print(f"Using vae scale: {vae_scale} and shift {vae_shift}")

    print("Loading dataset...")
    dataset = ImageDataset(
        root_dir=args.data_dir,
        load_into_ram=args.load_into_ram,
        dtype=torch.float32,
        resize_dim=args.resize_dim,
        conditional=False,
        is_latents=False,
    )

    if len(dataset) == 0:
        print("No images found in the dataset.")
        return

    loader_workers = 0 if args.load_into_ram else args.num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(loader_workers > 0),
        drop_last=False,
    )

    autocast_enabled = True

    if args.show_sample:
        print("Generating sample grid...")
        os.makedirs("results", exist_ok=True)

        idx = random.randint(0, max(0, len(dataset) - args.batch_size))
        if args.load_into_ram:
            sample_tensors = dataset.tensors_list[idx : idx + args.batch_size]
        else:
            sample_tensors = []
            for i in range(idx, idx + args.batch_size):
                sample_tensors.append(dataset[i])

        sample_tensor = torch.stack(sample_tensors).to(device, dtype=dtype)

        with torch.no_grad():
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = vae.encode(sample_tensor)
                if hasattr(output, "latent_dist"):
                    latents = output.latent_dist.sample()
                else:
                    latents = output

                decoded = vae.decode(latents).sample

        # Denormalize from[-1, 1] to [0, 1] for saving
        original_imgs = (sample_tensor.float().cpu() + 1.0) / 2.0
        decoded_imgs = (decoded.float().cpu() + 1.0) / 2.0

        original_imgs = original_imgs.clamp(0, 1)
        decoded_imgs = decoded_imgs.clamp(0, 1)

        comparison = torch.cat([original_imgs, decoded_imgs], dim=0)

        save_path = os.path.join("results", "vae_sample_grid.png")
        torchvision.utils.save_image(comparison, save_path, nrow=len(sample_tensors))
        print(f"Saved sample grid to {save_path}")

    if args.show_encoded:
        print("Generating sample grid...")
        os.makedirs("results", exist_ok=True)
        dataset = ImageDataset(
            root_dir=args.data_dir,
            load_into_ram=args.load_into_ram,
            dtype=torch.float32,
            resize_dim=args.resize_dim,
            conditional=False,
            is_latents=True,
            vae_scale=vae_scale,
            vae_shift=vae_shift,
            compute_normalization=False,
        )
        idx = random.randint(0, max(0, len(dataset) - args.batch_size))
        if args.load_into_ram:
            sample_tensors = dataset.tensors_list[idx : idx + args.batch_size]
        else:
            sample_tensors = []
            for i in range(idx, idx + args.batch_size):
                sample_tensors.append(dataset[i])
        sample_tensor = torch.stack(sample_tensors).to(device, dtype=dtype)

        with torch.no_grad():
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                decoded = vae.decode(sample_tensor).sample
                decoded = decoded.float().clamp(-1, 1)

        # Denormalize from[-1, 1] to [0, 1] for saving
        original_imgs = (sample_tensor.float().cpu() + 1.0) / 2.0
        decoded_imgs = (decoded.float().cpu() + 1.0) / 2.0

        original_imgs = original_imgs.clamp(0, 1)
        decoded_imgs = decoded_imgs.clamp(0, 1)

        comparison = torch.cat([decoded_imgs], dim=0)

        save_path = os.path.join("results", "vae_encoded_grid.png")
        torchvision.utils.save_image(comparison, save_path, nrow=len(sample_tensors))
        print(f"Saved sample grid to {save_path}")

    processed_count = 0
    for batch_tensor in tqdm(dataloader, desc="Encoding to latents"):
        batch_tensor = batch_tensor.to(device, dtype=dtype)
        with torch.no_grad():
            with torch.autocast(
                device_type=device,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = vae.encode(batch_tensor)
                if hasattr(output, "latent_dist"):
                    latents = output.latent_dist.sample()
                elif isinstance(output, tuple):
                    latents = (
                        output[0].sample()
                        if hasattr(output[0], "sample")
                        else output[0]
                    )
                else:
                    latents = output

        latents = latents.cpu().float()
        if args.half:
            latents = latents.to(torch.float16)
        latents = latents.numpy()

        # Retrieve paths matching the exact items in the current batch
        batch_size_actual = batch_tensor.shape[0]
        batch_paths = dataset.img_paths[
            processed_count : processed_count + batch_size_actual
        ]
        processed_count += batch_size_actual

        for latent, p in zip(latents, batch_paths):
            npz_path = p.with_suffix(".npz")
            np.savez(npz_path, latent=latent)


if __name__ == "__main__":
    main()
