import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from omegaconf import OmegaConf
import logging

from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.utils.visualization import (
    visualize_flow_matching,
)
from toy_diffusion.trainer import Trainer
from toy_diffusion.utils.logging_utils import Logger


def run_manifold(args):
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

    save_dir = "results/manifold_exp"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"loss_{config['loss_target']}_dataset_{config['dataset_type']}_path_{config['schedule_type']}_{config['model_type']}",
    )
    logging.info(cfg)

    logging.info(f"--- Running Experiment: {cfg.experiment.name} ---")
    logging.info(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    Ds = [2, 8, 16, 512]
    targets = ["x", "eps", "v"]

    logging.info(f"Running Experiment on {config['device']}")
    logging.info(f"Loss Space: {config['loss_target']} | Dimensions: {Ds}")

    fig, axes = plt.subplots(len(Ds), 4, figsize=(12, 3 * len(Ds)))

    cols = ["Ground Truth", "x-pred", "eps-pred", "v-pred"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight="bold")

    # for asuka dataset we need a lot of samples
    num_sampling = 2000
    if config["dataset_type"] in ["nerv", "asuka"]:
        num_sampling = num_sampling * 10

    for i, D in enumerate(Ds):
        logging.info(f"\n--- Processing Dimension D={D} ---")
        config["projection_dim"] = D

        dataset = SyntheticDataset(
            name=config["dataset_type"],
            n_samples=config["n_samples"],
            projection_dim=D,
            image_path=config["image_path"],
            font_path=config["font_path"],
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        num_sampling = min(num_sampling, len(dataset))

        gt_high = dataset.data[:num_sampling]
        gt_2d = gt_high @ dataset.P

        ax_gt = axes[i, 0]
        ax_gt.scatter(gt_2d[:, 0], gt_2d[:, 1], s=0.5, c="orange")
        ax_gt.set_ylabel(f"D={D}", fontsize=12, fontweight="bold")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        ax_gt.set_aspect("equal")

        for j, pred_target in enumerate(targets):
            logging.info(f"Training {pred_target}-prediction...")

            trainer = Trainer(config, prediction_target=pred_target, dataset=dataset)

            trainer.train(
                config["epochs"],
                dataloader,
                log_interval=config["epochs"],
            )
            samples, traj = trainer.sample(
                pred_target,
                num_sampling,
            )
            ax = axes[i, j + 1]
            ax.scatter(samples[:, 0], samples[:, 1], s=0.5, c="blue")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)

            if D == 2:
                logging.info(
                    f"  Generating Flow Visualization for D={D}, {pred_target}..."
                )
                traj_save_path = f"{save_dir}/path_{config['schedule_type']}_vis_D{D}_{pred_target}_{config['dataset_type']}.png"
                visualize_flow_matching(
                    trainer, dataset, D, pred_target, traj_save_path, num_sampling
                )

    os.makedirs("results", exist_ok=True)
    save_path = f"{save_dir}/full_experiment_loss_{config['loss_target']}_dataset_{config['dataset_type']}_path_{config['schedule_type']}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logging.info(f"\nExperiment Complete. Result saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy Diffusion Training Script")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to the YAML configuration file",
    )

    # Optional: Allow overriding specific config keys from CLI
    # Example: python train.py --config conf.yaml training.lr=0.0001
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (e.g., training.lr=1e-4)",
    )

    args = parser.parse_args()
    os.makedirs("results/manifold_exp", exist_ok=True)
    run_manifold(args)
