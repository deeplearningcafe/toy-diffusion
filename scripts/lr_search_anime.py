import torch
import numpy as np
import argparse
import os
import logging
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm

from toy_diffusion.trainer import Trainer

from toy_diffusion.data.image import ImageDataset

from toy_diffusion.utils.visualization import visualize_lr_search

from toy_diffusion.utils.logging_utils import Logger


def run_lr_search(args):
    """
    Performs a Learning Rate Range Test (Leslie Smith) to find the optimal
    learning rate for the anime faces dataset.
    """
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
        "device": device,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"results/lr_search/{timestamp}"
    Logger.setup_logging(save_dir=save_dir, logging_name="lr_search")
    logging.info(f"Starting LR Search with config: {config}")

    dataset = ImageDataset(
        root_dir=config["data_path"], num_workers=config["num_workers"]
    )

    batch_size = config.get("batch_size", 64)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    config["use_scheduler"] = False
    trainer = Trainer(
        config,
        prediction_target=config["loss_target"],
        dataset=dataset,
        autocast_dtype=autocast_dtype,
    )

    start_lr = args.start_lr
    end_lr = args.end_lr
    num_steps = args.num_steps

    # Calculate the multiplicative factor: lr_t = start_lr * (gamma ^ t)
    gamma = (end_lr / start_lr) ** (1 / num_steps)

    logging.info(
        f"Running LR Range Test from {start_lr:.2e} to {end_lr:.2e} over {num_steps} steps."
    )

    lrs = []
    losses = []
    current_lr = start_lr

    for param_group in trainer.optimizer.param_groups:
        param_group["lr"] = current_lr

    trainer.model.train()
    data_iter = iter(dataloader)

    progress_bar = tqdm(range(num_steps), desc="LR Search")

    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = current_lr

        loss_tensor = trainer.train_step(batch)
        loss = loss_tensor.item()

        if (
            np.isnan(loss)
            or np.isinf(loss)
            or (len(losses) > 0 and loss > 4 * min(losses))
        ):
            logging.info(
                f"Loss diverged at step {step}, LR={current_lr:.2e}, Loss={loss:.4f}"
            )
            break

        lrs.append(current_lr)
        losses.append(loss)

        current_lr *= gamma

        progress_bar.set_postfix({"lr": f"{current_lr:.2e}", "loss": f"{loss:.4f}"})

    # Simple moving average smoothing
    window_size = 20
    if len(losses) > window_size:
        smoothed_losses = np.convolve(
            losses, np.ones(window_size) / window_size, mode="valid"
        )
        pad_width = len(losses) - len(smoothed_losses)
        smoothed_losses = np.pad(smoothed_losses, (pad_width, 0), mode="edge")
    else:
        smoothed_losses = np.array(losses)

    # Find steepest descent (minimum gradient of the smoothed loss)
    min_loss_idx = np.argmin(smoothed_losses)
    min_loss_lr = lrs[min_loss_idx]

    gradients = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(gradients)
    steepest_lr = lrs[steepest_idx]

    suggested_lr = steepest_lr

    logging.info(f"LR Search Complete.")
    logging.info(f"Min Loss LR: {min_loss_lr:.2e}")
    logging.info(f"Steepest Descent LR: {steepest_lr:.2e}")
    logging.info(f"Suggested LR: {suggested_lr:.2e}")

    save_path = f"{save_dir}/lr_range_test.png"
    visualize_lr_search(lrs, losses, smoothed_losses, suggested_lr, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument(
        "--start_lr", type=float, default=1e-7, help="Starting learning rate"
    )
    parser.add_argument(
        "--end_lr", type=float, default=1e-3, help="Ending learning rate"
    )
    parser.add_argument(
        "--num_steps", type=int, default=1000, help="Number of steps for the range test"
    )
    os.makedirs("results/lr_search", exist_ok=True)
    run_lr_search(args)
