import argparse
import os
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from diffusers import AutoencoderKL
from datetime import datetime

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.image import ImageDataset
from toy_diffusion.ui.inference_app import (
    create_inference_ui,
    generate_images_custom,
)

def main():
    parser = argparse.ArgumentParser(
        description="Run CLI or GUI inference for diffusion models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="1girl, solo, asuka langley, neon genesis evangelion, "
                "red hair, blue eyes",
        help="Prompt for conditional generation",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="Negative prompt for conditional generation",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=6.0,
        help="Classifier-Free Guidance (CFG) scale",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Gradio web UI instead of running CLI inference",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Override config parameters",
    )

    args = parser.parse_args()

    # Load configuration
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.checkpoint_dir is not None:
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

    # Setup seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize VAE parameters
    vae_scale = 1.0
    vae_shift = 0.0
    if config.get("is_latents", False) and "vae_pretrained" in config:
        vae_config = AutoencoderKL.load_config(config["vae_pretrained"])
        vae_scale = vae_config.get("scaling_factor", 1.0)
        vae_shift = vae_config.get("shift_factor", 0.0)
        if vae_shift is None:
            vae_shift = 0.0

    # Load dataset to extract vocabulary
    print(f"Loading dataset from {config.get('data_path')}...")
    dataset = ImageDataset(
        root_dir=config.get("data_path"),
        num_workers=config.get("num_workers", 4),
        resize_dim=config.get("resize_dim", None),
        conditional=config.get("is_conditional", False),
        is_latents=config.get("is_latents", False),
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        compute_normalization=config.get("compute_normalization", False),
        exclude_tags=config.get("exclude_tags", []),
        is_finetune=config.get("is_finetune", False),
        finetune_orig_ratio=config.get("finetune_orig_ratio", 0.05),
        load_into_ram=config.get("load_into_ram", True),
    )

    config["vae_scale"] = dataset.vae_scale
    config["vae_shift"] = dataset.vae_shift

    pred_target = config.get("prediction_target", config.get("loss_target", "v"))
    if config.get("is_latents"):
        is_conditional = config.get("is_conditional", False)
        config["in_channels"] = (
            dataset[0][0].shape[0] if is_conditional else dataset[0].shape[0]
        )

    # loads checkpoint weights
    trainer = Trainer(
        config=config,
        dataset=dataset,
        prediction_target=pred_target,
    )

    output_dir = f"results/inference/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    if args.gui:
        print(f"Launching Gradio app on port {args.port}...")
        demo = create_inference_ui(
            trainer=trainer,
            default_prompt=args.prompt,
            default_neg_prompt=args.neg_prompt,
            default_steps=args.steps,
            default_cfg=args.cfg,
            default_batch_size=args.batch_size,
            output_dir=output_dir,
        )
        demo.launch(server_port=args.port, share=False)
    else:
        print("Running CLI inference...")
        images = generate_images_custom(
            trainer=trainer,
            prompt=args.prompt,
            neg_prompt=args.neg_prompt,
            num_samples=args.num_samples,
            steps=args.steps,
            cfg_scale=args.cfg,
            batch_size=args.batch_size,
            seed=seed,
        )
        
        # Save outputs
        grid_img = images[0]
        grid_img.save(f"{output_dir}/grid_output.png")
        print(f"Saved grid output to {output_dir}/grid_output.png")
        
        for idx, img in enumerate(images[1:]):
            img_path = f"{output_dir}/sample_{idx}.png"
            img.save(img_path)
            print(f"Saved sample {idx} to {img_path}")

if __name__ == "__main__":
    main()