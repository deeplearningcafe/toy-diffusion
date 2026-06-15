import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
from datetime import datetime
from diffusers import AutoencoderKL
import random
from PIL import Image

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.image import ImageDataset, TieredBatchSampler
from toy_diffusion.utils.visualization import (
    visualize_image_grid,
)
from toy_diffusion.utils.logging_utils import Logger
from toy_diffusion.utils.evaluation_utils import evaluate_model


def run_overfit_batch_experiment(args):
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
    random.seed(cfg.experiment.seed)

    config.setdefault("perturb_t", 0.5)
    config.setdefault("perturb_scale", 0.4)

    D = 2
    config["projection_dim"] = D
    batch_size = config["batch_size"]

    # update conf for overfit batch
    # 5k steps
    config["epochs"] = 5000
    config["sample_interval"] = 500
    config["save_interval"] = 200000

    logging.info("\n=== Step 1: Pretraining Base FM Model (1-Rectified Flow) ===")

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
    # here we just get a subset of 1 batch

    # We look at the tiers already calculated during dataset init
    valid_tiers = [
        t_idx_list
        for t_idx_list in dataset.tiers.values()
        if len(t_idx_list) >= batch_size
    ]

    if not valid_tiers:
        logging.warning(
            f"No single tier has {batch_size} samples. Sampling randomly from whole dataset."
        )
        indices = random.sample(range(len(dataset)), min(len(dataset), batch_size))
    else:
        chosen_tier_indices = random.choice(valid_tiers)
        indices = random.sample(chosen_tier_indices, batch_size)

    batch_dataset = dataset
    batch_dataset.tensors_list = [batch_dataset.tensors_list[i] for i in indices]
    batch_dataset.img_paths = [batch_dataset.img_paths[i] for i in indices]

    if is_conditional:
        batch_dataset._build_vocab()

    config["vae_scale"] = batch_dataset.vae_scale
    config["vae_shift"] = batch_dataset.vae_shift
    print(
        f"Using vae scale: {batch_dataset.vae_scale} and shift {batch_dataset.vae_shift}"
    )

    if is_conditional and hasattr(batch_dataset, "tiers"):
        batch_sampler = TieredBatchSampler(
            batch_dataset.tiers, batch_size, drop_last=True
        )
        dataloader = DataLoader(
            batch_dataset,
            batch_sampler=batch_sampler,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            batch_dataset,
            batch_size=batch_size,
            num_workers=config["num_workers"],
            persistent_workers=True if config["num_workers"] > 0 else False,
            shuffle=True,
            pin_memory=True,
        )

    if config.get("is_latents"):
        config["in_channels"] = (
            batch_dataset[0][0].shape[0]
            if is_conditional
            else batch_dataset[0].shape[0]
        )
        print(f"Using in_channels: {config['in_channels']}")

    data_shape = list(
        batch_dataset[0][0].shape if is_conditional else batch_dataset[0].shape
    )
    logging.info(f"Image Shape: {data_shape}")

    pred_target = config["loss_target"]
    trainer = Trainer(config, prediction_target=pred_target, dataset=batch_dataset)
    # we need to change the prompts to use the ones sampled
    # TODO: implement a method in the dataset to return prompts, not using the tensors_list
    prompts = [{"prompt": data[-1]} for data in batch_dataset.tensors_list]
    trainer.sample_configs = prompts

    logging.info("Generating ground truth grid from original image paths...")
    gt_images = []
    # the sampling code uses grids of 16
    for p in batch_dataset.img_paths[:16]:
        try:
            img = Image.open(p)
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    baimg = Image.new("RGB", img.size, (255, 255, 255))
                    baimg.paste(img, (0, 0), img)
                    img = baimg
                else:
                    img = img.convert("RGB")

            if config.get("resize_dim") is not None:
                resample_filter = getattr(Image, "Resampling", Image).BILINEAR
                img = img.resize(
                    (config["resize_dim"], config["resize_dim"]),
                    resample=resample_filter,
                )

            img_np = (np.array(img).astype(np.float32) / 127.5) - 1.0
            gt_images.append(img_np)
        except Exception as e:
            logging.warning(f"Could not load image {p} for GT grid: {e}")

    if gt_images:
        gt_images_np = np.stack(gt_images)
        gt_save_path = os.path.join(save_dir, "ground_truth_grid.png")
        visualize_image_grid(
            gt_images_np, gt_save_path, nrow=int(np.ceil(np.sqrt(len(gt_images_np))))
        )
        logging.info(f"Saved ground truth grid to {gt_save_path}")

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
        f"overfit_batch_{config['schedule_type']}_{config['prediction_target']}",
        data_shape,
        fid_samples=512,
        batch_size=batch_size * 6,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs("results/images", exist_ok=True)
    run_overfit_batch_experiment(args)
