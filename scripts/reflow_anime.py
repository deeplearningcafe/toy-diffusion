import torch
import numpy as np
import argparse
import copy
import os
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
from datetime import datetime

from toy_diffusion.trainer import Trainer
from toy_diffusion.data.image import ImageDataset
from toy_diffusion.data.coupling import Coupling

from toy_diffusion.utils.logging_utils import Logger
from toy_diffusion.utils.evaluation_utils import evaluate_model


def run_reflow_anime_faces_experiment(args):
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
    save_dir = f"results/reflow-anime/{timestamp}"
    Logger.setup_logging(
        save_dir=save_dir,
        logging_name=f"reflow_anime_log",
    )
    logging.info(cfg)

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    batch_size = config["batch_size"]

    dataset = ImageDataset(
        root_dir=config["data_path"], num_workers=config["num_workers"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    data_shape = list(dataset[0].shape)
    logging.info(f"Image Shape: {data_shape}")

    fm_config = config.copy()
    fm_config["schedule_type"] = "linear"
    fm_config["loss_target"] = "v"

    ddpm_config = config.copy()
    ddpm_config["schedule_type"] = "ddpm"
    ddpm_config["loss_target"] = "eps"
    ddpm_config["timestep_sampling"] = "uniform"
    # using the lr search this are more optimal
    ddpm_config["lr"] = 2e-5

    ddpm_config["input_perturbation"] = 0.0

    ddpm_config["weight_fn_name"] = None

    logging.info("\n=== Step 1: Pretraining Base FM Model (1-Rectified Flow) ===")

    base_trainer = Trainer(fm_config, prediction_target="v", dataset=dataset)
    base_model = base_trainer.train(
        fm_config["epochs"],
        dataloader,
        log_interval=1,
        sample_interval=1,
        timestamp=timestamp,
    )

    evaluate_model(
        base_trainer,
        "v",
        save_dir,
        "1_base_fm",
        data_shape,
        fid_samples=3000,
        batch_size=batch_size * 6,
    )

    logging.info("\n=== Step 2: Generating Coupling (Noise -> Generated Data) ===")

    coupling_dataset = Coupling(
        model=base_model,
        schedule=base_trainer.schedule,
        num_samples=len(dataset) // 2,
        dim=tuple(data_shape),
        device=device,
        batch_size=batch_size * 6,
        autocast_dtype=base_trainer.autocast_dtype,
    )

    coupling_loader = DataLoader(
        coupling_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )
    torch.cuda.empty_cache()

    logging.info(
        "\n=== Step 3: Training Reflowed FM Model (2-Rectified Flow) with NFE=4 for sampling ==="
    )
    # low lr as the model is already trained
    fm_config["lr"] = fm_config["lr"] / 4

    fm_config["timestep_sampling"] = "uniform"

    reflow_trainer = Trainer(
        config=fm_config,
        dataset=coupling_dataset,
        prediction_target="v",
        pretrained_model=copy.deepcopy(base_model),
    )

    reflow_model = reflow_trainer.train(
        epochs=config["epochs"] // 2,
        dataloader=coupling_loader,
        log_interval=1,
        sample_interval=1,
        timestamp=timestamp,
        num_steps=4,  # use less steps for reflow
    )

    evaluate_model(
        reflow_trainer,
        "v",
        save_dir,
        "2_reflow_fm",
        data_shape,
        fid_samples=3000,
        batch_size=batch_size * 6,
        num_steps=4,
    )
    torch.cuda.empty_cache()

    logging.info("\n=== Step 4: Training DDPM Baseline ===")

    ddpm_trainer = Trainer(
        config=ddpm_config,
        dataset=dataset,
        prediction_target="eps",
    )

    ddpm_model = ddpm_trainer.train(
        epochs=config["epochs"],
        dataloader=dataloader,
        log_interval=1,
        sample_interval=1,
        timestamp=timestamp,
    )

    evaluate_model(
        ddpm_trainer,
        "eps",
        save_dir,
        "3_ddpm_base",
        data_shape,
        fid_samples=3000,
        batch_size=batch_size * 6,
    )
    torch.cuda.empty_cache()

    logging.info("\n=== Step 5: Generating Coupling using DDPM model ===")

    ddpm_coupling_dataset = Coupling(
        model=ddpm_model,
        schedule=ddpm_trainer.schedule,
        num_samples=len(dataset) // 2,
        dim=tuple(data_shape),
        device=device,
        batch_size=batch_size * 6,
        autocast_dtype=base_trainer.autocast_dtype,
    )

    ddpm_coupling_loader = DataLoader(
        ddpm_coupling_dataset,
        batch_size=batch_size,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        shuffle=True,
        pin_memory=True,
    )

    logging.info("\n=== Step 6: Training Reflowed DDPM Model (2-Rectified Flow) ===")

    ddpm_config["lr"] = ddpm_config["lr"] / 4
    ddpm_reflow_trainer = Trainer(
        config=ddpm_config,
        dataset=ddpm_coupling_dataset,
        prediction_target="eps",
        pretrained_model=copy.deepcopy(ddpm_model),
    )

    ddpm_reflow_model = ddpm_reflow_trainer.train(
        epochs=config["epochs"] // 2,
        dataloader=ddpm_coupling_loader,
        log_interval=1,
        sample_interval=1,
        timestamp=timestamp,
    )

    evaluate_model(
        ddpm_reflow_trainer,
        "eps",
        save_dir,
        "4_reflow_ddpm",
        data_shape,
        fid_samples=3000,
        batch_size=batch_size * 2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs("results/reflow-anime", exist_ok=True)
    run_reflow_anime_faces_experiment(args)
