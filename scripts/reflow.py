import torch
import numpy as np
import copy
import argparse
import os
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.data.coupling import Coupling
from toy_diffusion.utils.visualization import (
    visualize_flow_matching,
    visualize_finetune_comparison,
    visualize_step_comparison,
    visualize_cm_evolution,
)
from toy_diffusion.utils.logging_utils import Logger


def run_reflow_experiment(args):
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
    save_dir = "results/reflow_exp_test"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"loss_{config['loss_target']}_dataset_{config['dataset_type']}_path_{config['schedule_type']}_{config['model_type']}",
    )
    logging.info(cfg)

    config.setdefault("perturb_t", 0.5)
    config.setdefault("perturb_scale", 0.4)

    D = 2
    config["projection_dim"] = D
    batch_size = config["batch_size"]

    font_path = config.get("font_path", None)
    if font_path is None and os.path.exists(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    ):
        config["font_path"] = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    # Step 1: Pretrain Base FM Model (1-Rectified Flow)
    logging.info("\n=== Step 1: Pretraining Base FM Model (1-Rectified Flow) ===")

    dataset = SyntheticDataset(
        name=config["dataset_type"],
        n_samples=config["n_samples"],
        projection_dim=D,
        font_path=config["font_path"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    fm_config = config.copy()
    fm_config["schedule_type"] = "linear"
    fm_config["loss_target"] = "v"

    base_trainer = Trainer(
        config=fm_config,
        dataset=dataset,
        prediction_target="v",
        autocast_enabled=False if device == "cpu" else True,
    )
    base_model = base_trainer.train(epochs=config["epochs"], dataloader=dataloader)

    visualize_flow_matching(
        base_trainer,
        dataset,
        D=D,
        pred_target="v",
        save_path=f"{save_dir}/1_base_fm_{config['dataset_type']}.png",
        num_samples=2048,
    )

    logging.info("\n=== Step 2: Generating Coupling (Noise -> Generated Data) ===")

    # z0 ~ N(0, I), z1 = ODE(z0)
    coupling_dataset = Coupling(
        model=base_model,
        schedule=base_trainer.schedule,
        num_samples=config["n_samples"] // 2,
        dim=D,
        device=device,
        batch_size=batch_size * 2,
    )

    coupling_loader = DataLoader(
        coupling_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    logging.info("\n=== Step 3: Training Reflowed Model (2-Rectified Flow) ===")

    # Initialize new model
    reflow_trainer = Trainer(
        config=fm_config,
        dataset=coupling_dataset,
        prediction_target="v",
        autocast_enabled=False if device == "cpu" else True,
    )

    reflow_model = reflow_trainer.train(
        epochs=config["epochs"] // 2, dataloader=coupling_loader
    )

    visualize_flow_matching(
        reflow_trainer,
        dataset,
        D=D,
        pred_target="v",
        save_path=f"{save_dir}/2_reflow_fm_{config['dataset_type']}.png",
        num_samples=2048,
    )

    logging.info("\n=== Step 4: Training DDPM Baseline ===")

    ddpm_config = config.copy()
    ddpm_config["schedule_type"] = "ddpm"
    ddpm_config["loss_target"] = "eps"

    ddpm_trainer = Trainer(
        config=ddpm_config,
        dataset=dataset,
        prediction_target="eps",
        autocast_enabled=False if device == "cpu" else True,
    )

    ddpm_model = ddpm_trainer.train(epochs=config["epochs"], dataloader=dataloader)

    visualize_flow_matching(
        ddpm_trainer,
        dataset,
        D=D,
        pred_target="eps",
        save_path=f"{save_dir}/3_ddpm_{config['dataset_type']}.png",
        num_samples=2048,
        clip_prediction=False,
    )
    logging.info(
        "\n=== Step 4: Generating Coupling (Noise -> Generated Data) using DDPM model ==="
    )

    ddpm_coupling_dataset = Coupling(
        model=ddpm_model,
        schedule=ddpm_trainer.schedule,
        num_samples=config["n_samples"] // 2,
        dim=D,
        device=device,
        batch_size=batch_size * 2,
        clip_prediction=False,
    )

    ddpm_coupling_loader = DataLoader(
        ddpm_coupling_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    logging.info(
        "\n=== Step 5: Training Reflowed Model (2-Rectified Flow) using e-pred ==="
    )

    ddpm_reflow_trainer = Trainer(
        config=ddpm_config,
        dataset=ddpm_coupling_dataset,
        prediction_target="eps",
        autocast_enabled=False if device == "cpu" else True,
    )

    ddpm_reflow_model = ddpm_reflow_trainer.train(
        epochs=config["epochs"] // 2, dataloader=ddpm_coupling_loader
    )

    visualize_flow_matching(
        ddpm_reflow_trainer,
        dataset,
        D=D,
        pred_target="eps",
        save_path=f"{save_dir}/5_reflow_ddpm_{config['dataset_type']}.png",
        num_samples=2048,
        clip_prediction=False,
    )

    logging.info("\n=== Step 6: Training Consistency Model ===")

    cm_config = config.copy()
    cm_config["loss_target"] = "consistency"

    cm_trainer = Trainer(
        config=cm_config,
        dataset=dataset,
        prediction_target="x",
        autocast_enabled=False if device == "cpu" else True,
    )
    cm_model = cm_trainer.train(epochs=config["epochs"], dataloader=dataloader)
    visualize_cm_evolution(
        model=cm_model,
        dataset=dataset,
        save_path=f"{save_dir}/6_cm_{config['dataset_type']}.png",
        num_samples=2048 * 2,
        num_steps=4,
        device=device,
        dataset_name=config["dataset_type"],
    )
    logging.info("Generating Step Comparison Grid...")

    trainers_list = [
        ddpm_trainer,
        base_trainer,
        reflow_trainer,
        ddpm_reflow_trainer,
        cm_trainer,
    ]
    model_names = [
        "DDPM",
        "FM",
        "Reflow",
        "Reflow DDPM",
        "CM",
    ]

    steps_list = [1, 2, 4, 10, 50]

    visualize_step_comparison(
        trainers=trainers_list,
        steps_list=steps_list,
        dataset=dataset,
        save_path=f"{save_dir}/6_step_comparison_{config['dataset_type']}.png",
        num_samples=2048,
        row_labels=model_names,
    )
    logging.info(
        f"\n=== Step 6: Generating Step Comparison with injected noise at t={config['perturb_t']} ==="
    )

    visualize_step_comparison(
        trainers=trainers_list,
        steps_list=steps_list,
        dataset=dataset,
        save_path=f"{save_dir}/7_step_comparison_perturb_t_{config['perturb_t']}_scale_{config['perturb_scale']}_{config['dataset_type']}.png",
        num_samples=2048,
        row_labels=model_names,
        perturb_t=config["perturb_t"],
        perturb_scale=config["perturb_scale"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs("results/reflow_exp", exist_ok=True)
    run_reflow_experiment(args)
