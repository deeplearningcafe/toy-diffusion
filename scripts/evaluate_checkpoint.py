import argparse
import os
import glob
import torch
import numpy as np
import logging
from datetime import datetime
from omegaconf import OmegaConf
from diffusers import AutoencoderKL

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.image import ImageDataset
from toy_diffusion.utils.logging_utils import Logger
from toy_diffusion.utils.evaluation_utils import evaluate_model


def run_evaluation(args):
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)

    cfg.training.resume_from_checkpoint = args.checkpoint_dir

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        logging.info("Warning: CUDA requested but not available. Using CPU.")
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

    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    base_ckpt = os.path.normpath(args.checkpoint_dir)
    epoch_name = os.path.basename(base_ckpt)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"results/sampling/loss_{config['loss_target']}_{config['schedule_type']}_{epoch_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    Logger.setup_logging(save_dir=save_dir, logging_name="evaluation")
    logging.info(f"Starting evaluation for checkpoint: {args.checkpoint_dir}")

    vae_scale = 1.0
    vae_shift = 0.0
    if config.get("is_latents", False) and "vae_pretrained" in config:
        vae_config = AutoencoderKL.load_config(config["vae_pretrained"])
        vae_scale = vae_config.get("scaling_factor", 1.0)
        vae_shift = vae_config.get("shift_factor", 0.0)
        if vae_shift is None:
            vae_shift = 0.0

    logging.info(f"Loading dataset from {config.get('data_path')}...")
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
    )

    config["vae_scale"] = dataset.vae_scale
    config["vae_shift"] = dataset.vae_shift
    print(f"Using vae scale: {dataset.vae_scale} and shift {dataset.vae_shift}")

    is_conditional = config.get("is_conditional", False)

    data_shape = list(dataset[0][0].shape if is_conditional else dataset[0].shape)
    logging.info(f"Image Shape: {data_shape}")

    prompts = None
    if is_conditional:
        logging.info(
            f"Extracting {args.num_samples} prompts from dataset for conditional evaluation..."
        )
        prompts = []
        num_to_extract = min(args.num_samples, len(dataset))
        for i in range(num_to_extract):
            # TODO: clean this , tensors_list shouldn't be accessed out of the class
            item = dataset.tensors_list[i]
            prompts.append(item[1])

        if len(prompts) < args.num_samples:
            repeats = (args.num_samples // len(prompts)) + 1
            prompts = (prompts * repeats)[: args.num_samples]

    config["use_scheduler"] = False
    pred_target = config.get("prediction_target", config.get("loss_target", "v"))

    if config.get("is_latents"):
        config["in_channels"] = (
            dataset[0][0].shape[0] if is_conditional else dataset[0].shape[0]
        )
        print(f"Using in_channels: {config['in_channels']}")

    # The Trainer automatically loads the checkpoint
    trainer = Trainer(
        config=config,
        dataset=dataset,
        prediction_target=pred_target,
    )

    if is_conditional and "text_enc" in trainer.model:
        text_enc = getattr(
            trainer.model["text_enc"], "_orig_mod", trainer.model["text_enc"]
        )
        text_enc.shuffle = False
        text_enc.cfg_dropout_prob = 0.0
        text_enc.tag_dropout_prob = 0.0

    cfg_scale = config.get("cfg_scale", 1.0)
    trainer.config["cfg_scale"] = cfg_scale
    logging.info(f"Using CFG Scale: {cfg_scale}")

    evaluate_model(
        trainer=trainer,
        pred_target=pred_target,
        save_dir=save_dir,
        prefix=epoch_name,
        data_shape=data_shape,
        fid_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        vae_batch_size=args.vae_batch_size,
        prompts=prompts,
        filter_type=args.filter_type,
        keep_ratio=args.keep_ratio,
        var_timesteps=args.var_timesteps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Checkpoint FID.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint folder (e.g., results/checkpoints/epoch_10)",
    )
    """ parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the original dataset to compute FID against"
    ) """
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3000,
        help="Number of samples to generate for FID",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generation and FID computation",
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--vae_batch_size",
        type=int,
        default=64,
        help="Batch size for VAE decoding to prevent OOM",
    )

    parser.add_argument(
        "--filter_type",
        type=str,
        default="none",
        help="Type of filter to apply: none, variance, random",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=1.0,
        help="Ratio of samples to keep after filtering",
    )
    parser.add_argument(
        "--var_timesteps",
        type=int,
        default=15,
        help="Number of timesteps to use for variance calculation",
    )

    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Override config parameters"
    )

    args = parser.parse_args()
    run_evaluation(args)
