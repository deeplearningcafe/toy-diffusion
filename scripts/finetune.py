import torch
import numpy as np
import copy
import os
import logging
from torch.utils.data import DataLoader
import argparse
from omegaconf import OmegaConf

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.utils.visualization import (
    visualize_flow_matching,
    visualize_finetune_comparison,
)
from toy_diffusion.utils.logging_utils import Logger


def run_finetune(args):
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
    save_dir = "results/finetune_exp"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"loss_{config['loss_target']}_dataset_{config['dataset_type']}_path_{config['schedule_type']}_{config['model_type']}",
    )
    logging.info(cfg)

    logging.info(f"--- Running Experiment: {cfg.experiment.name} ---")
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    batch_size = config["batch_size"]
    D = 2
    config["projection_dim"] = D

    font_path = config.get("font_path", None)
    if font_path is None:
        if os.path.exists("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            config["font_path"] = (
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            )
        else:
            logging.info("WARNING: No font_path provided. Kanji generation might fail.")

    logging.info(f"=== Step 1: Pretraining DDPM on {config['dataset_type']} ===")

    pretrain_dataset = SyntheticDataset(
        name=config["dataset_type"],
        n_samples=config["n_samples"],
        projection_dim=D,
        font_path=config["font_path"],
    )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)

    pretrain_config = config.copy()
    pretrain_config["schedule_type"] = "ddpm"
    pretrain_config["loss_target"] = "eps"

    pretrainer = Trainer(
        config=pretrain_config, dataset=pretrain_dataset, prediction_target="eps"
    )

    pretrained_model = pretrainer.train(
        epochs=config["epochs"],
        dataloader=pretrain_loader,
    )

    visualize_flow_matching(
        pretrainer,
        pretrain_dataset,
        D=2,
        pred_target="eps",
        save_path=f"{save_dir}/1_pretrain_ddpm_{config['dataset_type']}.png",
        num_samples=2048,
    )

    pretrained_model_state = copy.deepcopy(pretrained_model.state_dict())

    finetune_dataset = SyntheticDataset(
        name=f"{config['dataset_type']}_finetune",
        n_samples=config["n_samples"] // 4,
        projection_dim=D,
        font_path=config["font_path"],
    )
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)

    logging.info(
        f"\n=== Step 3: Finetuning DDPM (Target: eps) on Collapsed {config['dataset_type']} ==="
    )

    ddpm_model = copy.deepcopy(pretrained_model)
    ddpm_model.load_state_dict(pretrained_model_state)

    ddpm_finetuner = Trainer(
        config=pretrain_config,
        dataset=finetune_dataset,
        prediction_target="eps",
        pretrained_model=ddpm_model,
    )

    ddpm_finetuner.train(
        epochs=config["epochs"] // 2,
        dataloader=finetune_loader,
    )

    visualize_flow_matching(
        ddpm_finetuner,
        finetune_dataset,
        D=2,
        pred_target="eps",
        save_path=f"{save_dir}/2_finetune_ddpm_full_{config['dataset_type']}.png",
        num_samples=2048,
    )

    samples_ddpm, traj_ddpm = ddpm_finetuner.sample(pred_target="eps", num_sampling=500)
    visualize_finetune_comparison(
        samples_ddpm,
        traj_ddpm,
        finetune_dataset,
        save_path=f"{save_dir}/2_finetune_ddpm_traj_{config['dataset_type']}.png",
        title="Finetuned DDPM (SDE)",
    )

    logging.info(
        "\n=== Step 4: Finetuning Flow Matching (Target: v) on Collapsed GMM ==="
    )

    fm_config = config.copy()
    fm_config["schedule_type"] = "linear"
    fm_config["loss_target"] = "v"

    fm_model = copy.deepcopy(pretrained_model)
    fm_model.load_state_dict(pretrained_model_state)

    fm_finetuner = Trainer(
        config=fm_config,
        dataset=finetune_dataset,
        prediction_target="v",
        pretrained_model=fm_model,
    )

    fm_finetuner.train(epochs=config["epochs"] // 2, dataloader=finetune_loader)

    visualize_flow_matching(
        fm_finetuner,
        finetune_dataset,
        D=2,
        pred_target="v",
        save_path=f"{save_dir}/3_finetune_fm_full_{config['dataset_type']}.png",
        num_samples=2058,
    )

    samples_fm, traj_fm = fm_finetuner.sample(pred_target="v", num_sampling=500)
    visualize_finetune_comparison(
        samples_fm,
        traj_fm,
        finetune_dataset,
        save_path=f"{save_dir}/3_finetune_fm_traj_{config['dataset_type']}.png",
        title="Finetuned Flow Matching (ODE)",
    )

    logging.info(f"\nExperiment Completed. Check '{save_dir}' for visualizations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toy Diffusion Training Script")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to the YAML configuration file",
    )

    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Modify config options from command line (e.g., training.lr=1e-4)",
    )

    args = parser.parse_args()
    os.makedirs("results/finetune_exp", exist_ok=True)
    run_finetune(args)
