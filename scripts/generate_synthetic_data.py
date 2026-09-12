import argparse
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from tqdm import tqdm

from toy_diffusion.data.image import ImageDataset
from toy_diffusion.trainer import Trainer


def save_single_sample(
    img_np: np.ndarray,
    prompt: str,
    out_img_path: Path,
    out_txt_path: Path,
    quality: int = 80,
    speed: int = 6,
):
    """Encodes an RGB array to AVIF and writes the prompt atomically."""
    # Denormalize [-1.0, 1.0] float32 image to [0, 255] uint8
    img_uint8 = np.clip((img_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="RGB")

    tmp_id = uuid.uuid4().hex[:8]
    tmp_img = out_img_path.with_name(f"{out_img_path.name}_{tmp_id}.tmp")
    tmp_txt = out_txt_path.with_name(f"{out_txt_path.name}_{tmp_id}.tmp")

    try:
        pil_img.save(tmp_img, "AVIF", quality=quality, speed=speed)
        with open(tmp_txt, "w", encoding="utf-8") as f:
            f.write(prompt)

        tmp_img.rename(out_img_path)
        tmp_txt.rename(out_txt_path)
    finally:
        if tmp_img.exists():
            tmp_img.unlink()
        if tmp_txt.exists():
            tmp_txt.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic AVIF dataset for pixel adaptation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Base configuration file path.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing the pretrained latent model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory to save synthetic AVIF images and prompts.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=80000,
        help="Total number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DiT sampling.",
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=16,
        help="Batch size for VAE decoding.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of ODE sampling steps.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
        help="Classifier-Free Guidance scale.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=(
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, fewer digits, cropped, worst quality, low quality, "
            "normal quality, jpeg artifacts, signature, watermark, blurry"
        ),
        help="Negative prompt for CFG inference.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="AVIF quality parameter (1-100).",
    )
    parser.add_argument(
        "--avif_speed",
        type=int,
        default=6,
        help="AVIF compression speed (0=slowest, 10=fastest).",
    )
    parser.add_argument(
        "--io_workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel AVIF encoding.",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify configuration keys from command line.",
    )
    args = parser.parse_args()

    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)

    cfg.training.resume_from_checkpoint = args.checkpoint_dir
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config = {
        **OmegaConf.to_container(cfg.experiment),
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.training),
        **OmegaConf.to_container(cfg.diffusion),
        **OmegaConf.to_container(cfg.model),
        **OmegaConf.to_container(cfg.sampling),
        "device": device,
    }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vae_scale = config.get("vae_scale", 1.0)
    vae_shift = config.get("vae_shift", 0.0)

    print(f"Using vae scale: {vae_scale} and shift {vae_shift}")
    print(f"Scanning source dataset prompts from {config.get('data_path')}...")
    dataset = ImageDataset(
        root_dir=config.get("data_path"),
        num_workers=4,
        conditional=True,
        is_latents=config.get("is_latents", True),
        load_into_ram=False,
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        compute_normalization=False,
        use_short_prompts=config.get("use_short_prompts", False),
        tokenizer=config.get("hf_text_encoder", None),
    )

    config["vae_scale"] = dataset.vae_scale
    config["vae_shift"] = dataset.vae_shift
    config["use_scheduler"] = False
    config["ignore_checkpoint_optimizer"] = True

    sample_latent = dataset[0][0]
    data_shape = list(sample_latent.shape)
    config["in_channels"] = data_shape[0]

    pred_target = config.get("prediction_target", config.get("loss_target", "v"))
    trainer = Trainer(
        config=config,
        dataset=dataset,
        prediction_target=pred_target,
    )

    if "text_enc" in trainer.model:
        text_enc = getattr(
            trainer.model["text_enc"], "_orig_mod", trainer.model["text_enc"]
        )
        text_enc.shuffle = False
        text_enc.cfg_dropout_prob = 0.0
        text_enc.tag_dropout_prob = 0.0

    # Collect image paths and filter
    valid_tasks = []
    num_to_process = min(args.num_samples, len(dataset.img_paths))

    for idx in range(num_to_process):
        img_p = dataset.img_paths[idx]
        booru_stem = img_p.stem
        target_img = output_path / f"{booru_stem}_synthetic.avif"
        target_txt = output_path / f"{booru_stem}_synthetic.txt"

        if target_img.exists() and target_txt.exists():
            continue

        prompt_str = dataset._get_prompt(img_p)
        valid_tasks.append((prompt_str, target_img, target_txt))

    print(
        f"Found {len(dataset.img_paths)} total samples. "
        f"Target: {num_to_process} | Remaining to generate: {len(valid_tasks)}"
    )

    if len(valid_tasks) == 0:
        print("All synthetic samples already generated.")
        return

    with ThreadPoolExecutor(max_workers=args.io_workers) as io_pool:
        pbar = tqdm(total=len(valid_tasks), desc="Generating synthetic AVIFs")

        for i in range(0, len(valid_tasks), args.batch_size):
            chunk = valid_tasks[i : i + args.batch_size]
            current_bs = len(chunk)
            batch_prompts = [item[0] for item in chunk]

            # decoded RGB numpy array [B, H, W, 3] in [-1, 1]
            samples_np = trainer.sample(
                pred_target=pred_target,
                num_sampling=current_bs,
                data_shape=data_shape,
                num_steps=args.num_steps,
                return_traj=False,
                vae_batch_size=args.vae_batch_size,
                prompts=batch_prompts,
                negative_prompt=args.negative_prompt,
                cfg_scale=args.cfg_scale,
            )

            for b in range(current_bs):
                prompt_txt = batch_prompts[b]
                out_img = chunk[b][1]
                out_txt = chunk[b][2]
                img_data = samples_np[b]

                io_pool.submit(
                    save_single_sample,
                    img_data,
                    prompt_txt,
                    out_img,
                    out_txt,
                    args.quality,
                    args.avif_speed,
                )

            pbar.update(current_bs)

        pbar.close()

    print(f"Generation completed. Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
