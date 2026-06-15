import torch
import numpy as np
import os
import argparse
import logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.utils.visualization import (
    visualize_cm_evolution,
    visualize_step_comparison,
)
from toy_diffusion.utils.logging_utils import Logger


def run_cm_experiment(args):
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

    save_dir = "results/cm_exp_test"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"cm_test_{config['model_type']}",
    )
    logging.info(cfg)

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    D = 2
    config["projection_dim"] = D
    batch_size = config["batch_size"]

    font_path = config.get("font_path", None)
    if font_path is None and os.path.exists(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ):
        config["font_path"] = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    datasets_to_test = [
        "gmm",
        "gmm_imbalanced",
        "gmm_extreme_imbalanced",
        "gmm_long_tail",
        "pinwheel",
        "spiral",
    ]
    nfe_list = [1, 2, 4]

    for ds_name in datasets_to_test:
        logging.info(f"\n=== Training Consistency Model on {ds_name} ===")

        dataset_name_lower = ds_name.lower()
        is_imbalanced = any(
            kw in dataset_name_lower for kw in ["imbalanced", "extreme", "long_tail"]
        )
        samples_multiple = 1 if not is_imbalanced else 2

        dataset = SyntheticDataset(
            name=ds_name,
            n_samples=config["n_samples"],
            projection_dim=D,
            font_path=config.get("font_path", None),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.get("num_workers", 0),
            persistent_workers=True if config.get("num_workers", 0) > 0 else False,
            shuffle=True,
            pin_memory=True,
        )

        cm_config = config.copy()
        cm_config["loss_target"] = "consistency"

        cm_trainer = Trainer(
            config=cm_config,
            dataset=dataset,
            prediction_target="x",
            autocast_enabled=False if device == "cpu" else True,
        )

        cm_model = cm_trainer.train(epochs=config["epochs"], dataloader=dataloader)

        logging.info("--- Training Denoising Diffusion GAN ---")
        ddgan_config = config.copy()
        ddgan_config["model_type"] = "ddgan"
        ddgan_config["ddgan_steps"] = 4
        ddgan_config["compile_model"] = False
        ddgan_trainer = Trainer(
            config=ddgan_config,
            dataset=dataset,
            prediction_target="x",
            autocast_enabled=False if device == "cpu" else True,
        )
        ddgan_model = ddgan_trainer.train(
            epochs=config["epochs"], dataloader=dataloader
        )

        logging.info(f"--- Evaluating on {ds_name} ---")
        for nfe in nfe_list:
            save_path = f"{save_dir}/{ds_name}_cm_evolution_nfe_{nfe}.png"
            visualize_cm_evolution(
                model=cm_model,
                dataset=dataset,
                save_path=save_path,
                num_samples=2048 * samples_multiple,
                num_steps=nfe,
                device=device,
                dataset_name=ds_name,
            )

        logging.info(f"--- Evaluating on {ds_name} ---")
        save_path = f"{save_dir}/{ds_name}_cm_vs_ddgan.png"

        visualize_step_comparison(
            trainers=[cm_trainer, ddgan_trainer],
            steps_list=nfe_list,
            dataset=dataset,
            save_path=save_path,
            num_samples=2048 * samples_multiple,
            row_labels=["Consistency Model", "DD-GAN"],
        )

    logging.info("\nConsistency Models Experiment Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs("results/cm_exp", exist_ok=True)
    run_cm_experiment(args)
