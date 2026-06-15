import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
from datetime import datetime
from diffusers import AutoencoderKL

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.image import ImageDataset, TieredBatchSampler
from toy_diffusion.utils.logging_utils import Logger
from toy_diffusion.utils.evaluation_utils import evaluate_model


def run_anime_faces_experiment(args):
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        logging.info("Warning: CUDA requested but not available. Using CPU.")
        device = "cpu"
    else:
        device = cfg.training.device

    config = {
        **OmegaConf.to_container(cfg.experiment),
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.training),
        **OmegaConf.to_container(cfg.diffusion),
        **OmegaConf.to_container(cfg.model),
        **OmegaConf.to_container(cfg.sampling),
        "device": device,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"results/images/{timestamp}"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"loss_{config['loss_target']}_path_{config['schedule_type']}",
    )
    logging.info(cfg)

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    config.setdefault("perturb_t", 0.5)
    config.setdefault("perturb_scale", 0.4)

    D = 2
    config["projection_dim"] = D
    batch_size = config["batch_size"]

    logging.info("\n=== Step 1: Pretraining Base FM Model (1-Rectified Flow) ===")

    # Load VAE Config to dynamically extract scale and shift
    vae_scale = 1.0
    vae_shift = 0.0
    if config.get("is_latents", False) and "vae_pretrained" in config:
        vae_config = AutoencoderKL.load_config(config["vae_pretrained"])
        vae_scale = vae_config.get("scaling_factor", 1.0)
        vae_shift = vae_config.get("shift_factor", 0.0)
        if vae_shift is None:
            vae_shift = 0.0

    is_conditional = config.get("is_conditional", False)
    dataset = dataset = ImageDataset(
        root_dir=config["data_path"],
        num_workers=config["num_workers"],
        resize_dim=config.get("resize_dim", None),
        conditional=is_conditional,
        is_latents=config.get("is_latents", False),
        vae_scale=vae_scale,
        vae_shift=vae_shift,
        compute_normalization=config.get("compute_normalization", False),
        shuffle_tags=config.get("shuffle_tags", False),
        cfg_dropout_prob=config.get("cfg_dropout_prob", 0.0),
        tag_dropout_prob=config.get("tag_dropout_prob", 0.0),
    )

    config["vae_scale"] = dataset.vae_scale
    config["vae_shift"] = dataset.vae_shift
    print(f"Using vae scale: {dataset.vae_scale} and shift {dataset.vae_shift}")

    if is_conditional and hasattr(dataset, "tiers"):
        batch_sampler = TieredBatchSampler(dataset.tiers, batch_size, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            shuffle=True,
            pin_memory=True,
        )

    # if is latents then infer channels from data
    if config.get("is_latents"):
        config["in_channels"] = (
            dataset[0][0].shape[0] if is_conditional else dataset[0].shape[0]
        )
        print(f"Using in_channels: {config['in_channels']}")

    data_shape = list(dataset[0][0].shape if is_conditional else dataset[0].shape)
    logging.info(f"Image Shape: {data_shape}")

    pred_target = config["loss_target"]
    trainer = Trainer(config, prediction_target=pred_target, dataset=dataset)

    trainer.train(
        config["epochs"],
        dataloader,
        log_interval=config["log_interval"],
        sample_interval=config["sample_interval"],
        timestamp=timestamp,
        save_interval=config["save_interval"],
    )

    evaluate_model(
        trainer,
        config["prediction_target"],
        save_dir,
        f"{config['schedule_type']}_{config['prediction_target']}",
        data_shape,
        fid_samples=3000,
        batch_size=batch_size * 6,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs("results/images", exist_ok=True)
    run_anime_faces_experiment(args)
