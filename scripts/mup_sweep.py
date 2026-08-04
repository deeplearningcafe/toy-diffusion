import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from toy_diffusion.data.image import ImageDataset
from toy_diffusion.trainer import Trainer


def register_coord_hooks(model):
    """
    Attaches forward hooks to linear/convolutional layers to capture
    mean absolute activation scale per step.
    """
    hooks = []
    act_data = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                mean_abs = torch.abs(output).mean().item()
                if name not in act_data:
                    act_data[name] = []
                act_data[name].append(mean_abs)

        return hook

    # Hook key linear and conv layers in the blocks
    for name, module in model.named_modules():
        if any(kw in name for kw in ["attn", "mlp", "proj_out"]):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                h = module.register_forward_hook(hook_fn(name))
                hooks.append(h)
    return hooks, act_data


def run_sweep_step(
    width, lr, init_std, mup_enabled, steps=1000, dataset=None, is_coord_check=False
):
    """
    Initializes the model and runs train steps via the Trainer class.
    If is_coord_check is True, logs activation scale statistics.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mup_base_width = 256

    config = {
        "model_type": "dual_stream",
        "hidden_dim": width,
        "depth": 4,  # Minimal blocks for quick verification
        "num_heads": 4,
        "lr": lr,
        "init_std": init_std,
        "mup_enabled": mup_enabled,
        "mup_base_width": mup_base_width,
        "mup_input_alpha": 1.0,
        "mup_output_alpha": 1.0,
        "device": device,
        "batch_size": 256,
        "loss_target": "v",
        "schedule_type": "linear",
        "is_conditional": False,
        "use_ema": False,
        "compile_model": False,
    }

    # Use dataset or construct fallback
    if dataset is None:

        class SyntheticDataset(Dataset):
            def __len__(self):
                return 10000

            def __getitem__(self, idx):
                return torch.randn(4, 16, 16)

        dataset = SyntheticDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if torch.cuda.is_available() else False,
    )

    trainer = Trainer(
        config=config,
        dataset=dataset,
        prediction_target="v",
    )

    act_data = {}
    hooks = []
    if is_coord_check:
        hooks, act_data = register_coord_hooks(trainer.model)

    step_count = 0
    total_loss = 0.0

    while step_count < steps:
        for batch in dataloader:
            if step_count >= steps:
                break
            loss = trainer.train_step(batch)
            total_loss += loss.item()
            step_count += 1

    # Remove hooks to avoid leaking memory
    for h in hooks:
        h.remove()

    avg_loss = total_loss / steps
    return avg_loss, act_data


def plot_coord_check(results_data, steps):
    """
    Saves visual coordinate alignment plot for mup verification.
    """
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (mode, data_dict) in enumerate(results_data.items()):
        ax = axes[idx]
        for width, layers_data in data_dict.items():
            for layer_name, values in layers_data.items():
                # Pick a representative layer (e.g. self-attn) to plot
                if "attn.to_out" in layer_name:
                    ax.plot(
                        range(len(values)),
                        values,
                        label=f"W={width} ({layer_name[:15]})",
                        alpha=0.8,
                    )
        ax.set_title(f"Activation Coords - {mode.upper()}", fontsize=14)
        ax.set_xlabel("Steps", fontsize=12)
        ax.set_ylabel("Mean Absolute Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig("results/mup_coord_check.png", dpi=150)
    plt.close()


def plot_hyperparam_sweep(sweep_results):
    """
    Plots learning rate vs loss across initialization variances.
    """
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (mup_status, widths_data) in enumerate(sweep_results.items()):
        ax = axes[idx]
        title = "muP Enabled" if mup_status else "Standard Param (SP)"
        for width, grid_data in widths_data.items():
            for init_std, lr_losses in grid_data.items():
                lrs = sorted(list(lr_losses.keys()))
                losses = [lr_losses[lr] for lr in lrs]
                ax.plot(
                    lrs,
                    losses,
                    marker="o",
                    label=f"W={width}, Init={init_std}",
                    alpha=0.8,
                )
        ax.set_xscale("log")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Learning Rate", fontsize=12)
        ax.set_ylabel("Loss (1000 Steps)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig("results/mup_hyperparam_sweep.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="muP Verification Suite")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_example.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--coord_check",
        action="store_true",
        help="Perform activation scale coordinate checks instead of sweep",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of steps per parameter combination",
    )
    args = parser.parse_args()

    # Load default configs
    config = {}
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
        config = OmegaConf.to_container(cfg.data)

    # Initialize Dataset
    dataset = None
    data_path = config.get("data_path", "")
    if os.path.exists(data_path):
        print(f"Loading standard dataset from {data_path}...")
        dataset = ImageDataset(
            root_dir=data_path,
            load_into_ram=config.get("load_into_ram", False),
            num_workers=config.get("num_workers", 4),
            resize_dim=config.get("resize_dim", None),
            conditional=config.get("is_conditional", False),
            is_latents=config.get("is_latents", False),
        )
    else:
        print("Data path not configured. Utilizing synthetic datasets.")

    widths = [128, 256, 512]

    if args.coord_check:
        print("\n================================================")
        print("Starting Coordinate Check Verification Run...")
        print("================================================")
        coord_steps = 15  # Short step count is sufficient for scale checks
        results = {"sp": {}, "mup": {}}

        for mup in [False, True]:
            mode_key = "mup" if mup else "sp"
            for w in widths:
                print(f"Running mode: {mode_key.upper()} | Width: {w}...")
                _, act_data = run_sweep_step(
                    width=w,
                    lr=1e-3,
                    init_std=0.02,
                    mup_enabled=mup,
                    steps=coord_steps,
                    dataset=dataset,
                    is_coord_check=True,
                )
                results[mode_key][w] = act_data

        plot_coord_check(results, coord_steps)
        print("\nCoordinate check finished! Plot saved to results/mup_coord_check.png")

    else:
        print("\n================================================")
        print(f"Starting Hyperparameter Sweep ({args.steps} Steps)...")
        print("================================================")

        lrs = [3e-4, 8e-4, 1e-3, 3e-3, 6e-3]
        init_stds = [0.01, 0.02]

        # Grid Results: {mup_enabled: {width: {init_std: {lr: loss}}}}
        sweep_results = {
            False: {w: {init: {} for init in init_stds} for w in widths},
            True: {w: {init: {} for init in init_stds} for w in widths},
        }

        for mup in [False, True]:
            for w in widths:
                for init in init_stds:
                    for lr in lrs:
                        print(
                            f"Training: muP={mup} | Width={w} | Init={init} | LR={lr}"
                        )
                        loss, _ = run_sweep_step(
                            width=w,
                            lr=lr,
                            init_std=init,
                            mup_enabled=mup,
                            steps=args.steps,
                            dataset=dataset,
                            is_coord_check=False,
                        )
                        sweep_results[mup][w][init][lr] = loss

        plot_hyperparam_sweep(sweep_results)
        print("\nSweep completed! Results saved to results/mup_hyperparam_sweep.png")


if __name__ == "__main__":
    main()
