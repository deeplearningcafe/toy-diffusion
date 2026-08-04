import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

from toy_diffusion.data.image import ImageDataset, TieredBatchSampler
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
    width,
    lr,
    init_std,
    mup_enabled,
    steps=1000,
    dataset=None,
    is_coord_check=False,
):
    """
    Initializes the model and runs train steps via the Trainer class.
    Steps the optimizer explicitly to ensure weights update.
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
        "is_conditional": True,
        "use_ema": False,
        "compile_model": False, #True if not is_coord_check else False,
        "cross_attention_dim": 256,
        "num_workers": 4,
        "epochs": 3,
        "is_latents": True,
        "vae_pretrained": "kaiyuyue/FLUX.2-dev-vae",
        "warmup": 0.001,
    }

    # Use dataset or construct fallback
    if dataset is None:

        class SyntheticDataset(Dataset):
            def __len__(self):
                return 10000

            def __getitem__(self, idx):
                return torch.randn(4, 16, 16)

        dataset = SyntheticDataset()

    is_conditional = True
    if config.get("is_latents"):
        config["in_channels"] = (
            dataset[0][0].shape[0] if is_conditional else dataset[0].shape[0]
        )

    batch_sampler = TieredBatchSampler(
        dataset.tiers, config["batch_size"], drop_last=True
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
        pin_memory=True,
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

            if trainer.scaler:
                if trainer.grad_clip > 0.0:
                    trainer.scaler.unscale_(trainer.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(), trainer.grad_clip
                    )
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                if trainer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(), trainer.grad_clip
                    )
                trainer.optimizer.step()

            trainer.optimizer.zero_grad(set_to_none=True)
            if trainer.scheduler is not None:
                trainer.scheduler.step()

            total_loss += loss.item()
            step_count += 1

    # Remove hooks to avoid leaking memory
    for h in hooks:
        h.remove()

    avg_loss = total_loss / steps
    return avg_loss, act_data


def plot_coord_check(results_data, steps):
    """Saves visual coordinate alignment plot for mup verification.

    Organized into 2 columns (SP, MUP) and 1 row per width dimension (e.g., 128,
    256, 512).
    """
    os.makedirs("results/mup", exist_ok=True)

    target_keywords = [
        "qkv_image",
        "w12",
        "proj_image",
        "w3",
        "proj_out",
    ]

    modes = list(results_data.keys()) 
    if not modes:
        return

    widths = sorted(
        list(
            {
                w
                for mode_data in results_data.values()
                if mode_data
                for w in mode_data.keys()
            }
        )
    )

    if not widths:
        return

    n_rows = len(widths)
    n_cols = len(modes)

    # Create subplot grid: 1 row per width, 1 column per mode
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16, 8.5 * n_rows), squeeze=False
    )


    for col_idx, mode in enumerate(modes):
        data_dict = results_data.get(mode, {})
        for row_idx, width in enumerate(widths):
            ax = axes[row_idx, col_idx]
            layers_data = data_dict.get(width, {})

            if layers_data:
                for layer_name, values in layers_data.items():
                    # Plot layers matching MMDiT layer keywords
                    if any(kw in layer_name for kw in target_keywords):
                        short_name = layer_name.replace("unet.", "")
                        ax.plot(
                            range(len(values)),
                            values,
                            label=short_name,
                            alpha=0.7,
                            linewidth=1.5,
                        )

            ax.set_title(
                f"Activation Coords - {mode.upper()} (Width {width})",
                fontsize=12,
            )
            ax.set_xlabel("Steps", fontsize=10)
            ax.set_ylabel("Mean Absolute Value", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    fontsize=7,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                )

    plt.tight_layout()
    plot_path = "results/mup/mup_coord_check.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hyperparam_sweep(sweep_results):
    """
    Plots learning rate vs loss across initialization variances.
    """
    os.makedirs("results/mup", exist_ok=True)
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
    plot_path = "results/mup/mup_hyperparam_sweep.png"
    plt.savefig(plot_path, dpi=150)
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

    config = {}
    if os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
        config = OmegaConf.to_container(cfg.data)

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

    widths = [256, 512]

    if args.coord_check:
        print("\n================================================")
        print("Starting Coordinate Check Verification Run...")
        print("================================================")
        coord_steps = 30  # Short step count is sufficient for scale checks
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
        print("\nCoordinate check finished! Plot saved to results/mup/mup_coord_check.png")

    else:
        print("\n================================================")
        print(f"Starting Hyperparameter Sweep ({args.steps} Steps)...")
        print("================================================")

        lrs = [4e-4, 8e-4, 1e-3, 4e-3, 8e-3]
        init_stds = [0.01, 0.02]

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
                        torch.cuda.empty_cache()
                        sweep_results[mup][w][init][lr] = loss

        plot_hyperparam_sweep(sweep_results)
        print("\nSweep completed! Results saved to results/mup/mup_hyperparam_sweep.png")


if __name__ == "__main__":
    main()