import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.data.image import ImageDataset

from toy_diffusion.utils.visualization import (
    visualize_flow_matching,
    visualize_image_grid,
    visualize_image_trajectory,
    visualize_path,
)
from toy_diffusion.trainer import Trainer
from toy_diffusion.ui.app import create_ui
from toy_diffusion.paths.scheduler import LinearSchedule, DDPMSchedule, VESchedule

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def visualize_paths():
    ds_spiral = SyntheticDataset(name="spiral", n_samples=50000, projection_dim=0)
    ds_gmm = SyntheticDataset(name="gmm", n_samples=50000, projection_dim=0)

    datasets_to_plot = [
        # (ds_gmm, "Gaussian Mixture"),
        (ds_spiral, "Spiral"),
    ]

    print("Visualizing Linear Schedule...")
    visualize_path(
        LinearSchedule, datasets_to_plot, save_path="results/path_vis_linear.png"
    )

    print("Visualizing DDPM Schedule...")
    visualize_path(
        DDPMSchedule, datasets_to_plot, save_path="results/path_vis_ddpm.png"
    )

    print("Visualizing Karras Schedule...")
    visualize_path(
        VESchedule, datasets_to_plot, save_path="results/path_vis_karras.png"
    )


def plot_all_synthetic_datasets():
    from toy_diffusion.utils.visualization import visualize_base_datasets

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    has_font = os.path.exists(font_path)

    print("Generating base synthetic datasets...")
    gmm = SyntheticDataset("gmm", 50000)
    gmm_imb = SyntheticDataset("gmm_imbalanced", 50000)
    gmm_lt = SyntheticDataset("gmm_long_tail", 50000)

    spiral = SyntheticDataset("spiral", 50000)
    pinwheel = SyntheticDataset("pinwheel", 50000)

    dataset_rows = [
        [
            (gmm, "GMM (Balanced)"),
            (gmm_imb, "GMM (Imbalanced)"),
            (gmm_lt, "GMM (Long Tail)"),
        ],
        [(spiral, "Spiral"), (pinwheel, "Pinwheel")],
    ]

    if has_font:
        kanji1 = SyntheticDataset("kanji", 50000, font_path=font_path)
        kanji2 = SyntheticDataset("kanji_finetune", 50000, font_path=font_path)
        dataset_rows.append(
            [(kanji1, "Kanji (Pretrain: a)"), (kanji2, "Kanji (Finetune: o)")]
        )
    else:
        print("Font not found, skipping Kanji datasets.")

    os.makedirs("results", exist_ok=True)
    save_path = "results/all_base_synthetic_datasets.png"
    visualize_base_datasets(dataset_rows, save_path)
    print(f"Saved base datasets plot to {save_path}")


if __name__ == "__main__":
    # visualize_paths()
    # plot_all_synthetic_datasets()
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    demo = create_ui(args)
    demo.launch()
