import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import concurrent.futures
from PIL import Image
from toy_diffusion.paths.sampling import generate_samples
from toy_diffusion.evaluation import (
    compute_precision_recall,
    compute_curvature,
    chamfer_distance,
)


def visualize_base_datasets(
    dataset_rows, save_path="results/base_datasets.png", num_samples=2048
):
    """
    Visualizes a grid of synthetic datasets using scatter plots for
    coherence and better visibility of imbalanced data.

    Args:
        dataset_rows: List of lists containing tuples of (dataset, title).
        save_path: Path to save the generated plot.
        num_samples: Number of points to scatter per plot.
    """
    n_rows = len(dataset_rows)
    n_cols = max(len(row) for row in dataset_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor("white")

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(dataset_rows[r]):
                ds, name = dataset_rows[r][c]

                # Extract data and project to 2D if needed
                data = ds.data[:num_samples]
                if hasattr(ds, "P") and ds.P is not None:
                    data = data @ ds.P

                # Dynamically calculate boundaries for long-tail support
                x_min, x_max = data[:, 0].min(), data[:, 0].max()
                y_min, y_max = data[:, 1].min(), data[:, 1].max()
                x_margin = max((x_max - x_min) * 0.1, 1.0)
                y_margin = max((y_max - y_min) * 0.1, 1.0)

                ax.scatter(
                    data[:, 0],
                    data[:, 1],
                    s=5,
                    c="royalblue",
                    alpha=0.6,
                    edgecolors="none",
                )

                ax.set_title(name, fontsize=28, fontweight="bold")
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")
                ax.set_facecolor("white")
            else:
                ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logging.info(f"Saved base dataset scatter visualization to {save_path}")


def visualize_flow_matching(
    trainer,
    dataset,
    D,
    pred_target,
    save_path,
    num_samples: int = 2048,
    num_timesteps: int = 50,
    clip_prediction: bool = False,
):
    """
    Visualizes the Learned Model ODE vs Ground Truth Conditional Paths.
    """
    device = trainer.device
    schedule = trainer.schedule
    schedule_name = schedule.get_scheduler_type()
    dataset_name = trainer.config.get("dataset_type", "unknown").lower()

    c_data = "royalblue"
    c_mid = "gold"
    c_noise = "tomato"
    c_traj = "darkslategray"
    c_bg = "lightgray"

    gt_high = dataset.data[:num_samples]
    if hasattr(dataset, "P") and dataset.P is not None:
        gt_2d = gt_high @ dataset.P
    else:
        gt_2d = gt_high

    is_imbalanced = any(
        kw in dataset_name for kw in ["imbalanced", "extreme", "long_tail"]
    )

    if is_imbalanced:
        x_min, x_max = gt_2d[:, 0].min(), gt_2d[:, 0].max()
        y_min, y_max = gt_2d[:, 1].min(), gt_2d[:, 1].max()

        # Add a 10% margin to the bounds
        x_margin = max((x_max - x_min) * 0.1, 1.0)
        y_margin = max((y_max - y_min) * 0.1, 1.0)

        x_bounds = [x_min - x_margin, x_max + x_margin]
        y_bounds = [y_min - y_margin, y_max + y_margin]
    else:
        scale = 3.0
        x_bounds = [-scale, scale]
        y_bounds = [-scale, scale]

    z_noise = torch.randn(num_samples, D, device=device)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    legend_size = 20
    markerscale = 2.0

    def plot_background(ax):
        ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=10, c=c_bg, alpha=0.35, label="Data")
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot 1: Ground-Truth Conditional Probability Path
    ax = axes[0, 0]
    ax.set_title("Ground-Truth Conditional Probability Path", fontsize=28)
    plot_background(ax)

    ts_vis = torch.linspace(0, 1, 20).to(device)
    alpha_vis, sigma_vis, _, _ = schedule.get_coefficients(ts_vis)
    alpha_vis = alpha_vis.view(-1, 1)
    sigma_vis = sigma_vis.view(-1, 1)

    gt_tensor = torch.from_numpy(gt_high).to(device)

    for i in range(min(20, num_samples)):
        # Path: alpha(t) * x_1 + sigma(t) * x_0
        # shape: (Steps, D)
        path = alpha_vis * gt_tensor[i].unsqueeze(0) + sigma_vis * z_noise[i].unsqueeze(
            0
        )
        path_2d = path.cpu().numpy() @ dataset.P
        ax.plot(path_2d[:, 0], path_2d[:, 1], c=c_traj, alpha=0.6, linewidth=1.5)

    time_steps = [0.0, 0.5, 1.0]
    time_colors = [c_noise, c_mid, c_data]

    for t_val, color in zip(time_steps, time_colors):
        t_tensor = torch.tensor([t_val], device=device)
        alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)

        z_t = alpha * gt_tensor + sigma * z_noise
        z_t_2d = z_t.cpu().numpy() @ dataset.P

        ax.scatter(
            z_t_2d[:, 0], z_t_2d[:, 1], s=15, alpha=0.6, c=color, label=f"t={t_val}"
        )

    ax.legend(prop={"size": legend_size}, loc="upper right", markerscale=markerscale)

    # Plot (0, 1): Ground-Truth Marginal Vector Fields
    ax = axes[0, 1]
    ax.set_title("Ground-Truth Marginal Velocity Fields", fontsize=28)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    plot_background(ax)

    grid_res = 15
    x_g = np.linspace(x_bounds[0], x_bounds[1], grid_res)
    y_g = np.linspace(y_bounds[0], y_bounds[1], grid_res)
    X_g, Y_g = np.meshgrid(x_g, y_g)
    grid_flat = np.stack([X_g.flatten(), Y_g.flatten()], axis=1)

    # Project grid back to High Dim (D) if necessary
    grid_tensor_2d = torch.from_numpy(grid_flat).float().to(device)
    grid_tensor_high = grid_tensor_2d @ torch.from_numpy(dataset.P.T).float().to(device)

    vf_subset_size = min(1000, num_samples)
    gt_subset = gt_tensor[:vf_subset_size]

    for t_val, color in zip(time_steps, time_colors):
        v_high = compute_marginal_vector_field(
            schedule, t_val, grid_tensor_high, gt_subset, device=device
        )

        v_2d = v_high.cpu().numpy() @ dataset.P

        # Normalize for visualization (Quiver)
        norms = np.linalg.norm(v_2d, axis=1, keepdims=True)
        v_norm = v_2d / (norms + 1e-5)

        ax.quiver(
            X_g,
            Y_g,
            v_norm[:, 0].reshape(grid_res, grid_res),
            v_norm[:, 1].reshape(grid_res, grid_res),
            scale=25,
            width=0.006,
            color=color,
            alpha=0.8,
            label=f"v(t={t_val})",
        )

    ax.legend(prop={"size": legend_size}, loc="upper right")

    # Run Learned Model ODE
    is_conditional = getattr(trainer, "conditional", False)
    embeddings, attention_mask = None, None
    if is_conditional:
        with torch.no_grad():
            embeddings, attention_mask = trainer.model["text_enc"]([""] * num_samples)

    extra_kwargs = {}
    if schedule_name == "ddpm":
        extra_kwargs["clip_prediction"] = clip_prediction

    samples_final, traj = generate_samples(
        model=trainer.model,
        schedule=schedule,
        x=z_noise,
        diffusion_type=schedule_name,
        prediction_target=pred_target,
        num_steps=num_timesteps,
        is_conditional=is_conditional,
        embeddings=embeddings,
        attention_mask=attention_mask,
        projection_matrix=dataset.P,
        return_traj=True,
        **extra_kwargs,
    )

    logging.info(f"\n--- Evaluation Metrics for {save_path} ---")

    # traj is (B, Steps, 2)
    curvature = compute_curvature(traj)
    logging.info(f"  > Path Curvature: {curvature:.4f} (1.0 = Straight)")

    # B. Trajectory Chamfer Distance
    t_grid = torch.linspace(0, 1, num_timesteps + 1, device=device)
    gt_traj_list = []

    # Use the same z_noise and gt_tensor (subset) used for plotting
    for t_val in t_grid:
        t_tensor = torch.tensor([t_val], device=device)
        alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)
        z_t = alpha * gt_tensor + sigma * z_noise
        if hasattr(dataset, "P") and dataset.P is not None:
            z_t_2d = z_t.cpu().numpy() @ dataset.P
        else:
            z_t_2d = z_t.cpu().numpy()
        gt_traj_list.append(z_t_2d)

    # Stack to (B, Steps, 2)
    gt_traj = np.stack(gt_traj_list, axis=1)

    chamfer_traj = chamfer_distance(traj, gt_traj, is_trajectory=True)
    logging.info(f"  > Trajectory Chamfer Distance: {chamfer_traj:.4f}")

    if "gmm" in dataset_name or "pinwheel" in dataset_name:
        k = 8 if dataset_name == "gmm" else 5
        precision, recall = compute_precision_recall(samples_final, gt_2d, k=k)
        logging.info(f"  > Manifold Precision: {precision:.4f}")
        logging.info(f"  > Manifold Recall:    {recall:.4f}")
    else:
        logging.info(f"  > Precision/Recall skipped for dataset: {dataset_name}")

    logging.info("------------------------------------------\n")

    # Plot 2: Samples from Learned Marginal ODE
    ax = axes[1, 0]
    ax.set_title("Samples from Learned Marginal ODE", fontsize=28)
    plot_background(ax)

    # Plot intermediate distributions
    indices = [0, num_timesteps // 2, num_timesteps]

    for idx, t_val, color in zip(indices, time_steps, time_colors):
        idx = min(idx, traj.shape[1] - 1)
        pts = traj[:, idx, :]
        ax.scatter(pts[:, 0], pts[:, 1], s=15, alpha=0.6, c=color, label=f"t={t_val}")

    ax.legend(prop={"size": legend_size}, loc="upper right", markerscale=markerscale)

    # Plot 3: Trajectories of Learned Marginal ODE
    ax = axes[1, 1]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=28)
    plot_background(ax)

    for i in range(min(20, num_samples)):
        ax.plot(traj[i, :, 0], traj[i, :, 1], c=c_traj, alpha=0.5)

    ax.scatter(traj[:20, 0, 0], traj[:20, 0, 1], c=c_noise, s=20, zorder=5)
    ax.scatter(traj[:20, -1, 0], traj[:20, -1, 1], c=c_data, s=20, zorder=5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    if dataset_name == "gmm_long_tail":
        save_path_hist = f"{save_path.split('.')[0]}_hist{save_path.split('.')[1]}"
        visualize_long_tail_histogram(samples_final, gt_2d, save_path_hist)


def visualize_finetune_comparison(
    samples_final, traj, dataset, save_path, title="Finetune Result"
):
    """
    Simplified visualization for finetuning results showing trajectories
    over the new ground truth distribution.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    gt_high = dataset.data
    if dataset.P is not None:
        gt_2d = gt_high @ dataset.P
    else:
        gt_2d = gt_high

    idx = np.random.choice(len(gt_2d), min(2000, len(gt_2d)), replace=False)
    ax.scatter(
        gt_2d[idx, 0],
        gt_2d[idx, 1],
        s=10,
        c="lightgray",
        alpha=0.3,
        label="Target Data",
    )

    num_vis = min(20, traj.shape[0])
    for i in range(num_vis):
        ax.plot(traj[i, :, 0], traj[i, :, 1], c="darkslategray", alpha=0.5)

    ax.scatter(
        traj[:num_vis, 0, 0],
        traj[:num_vis, 0, 1],
        c="tomato",
        s=20,
        label="Noise (t=0)",
    )
    ax.scatter(
        traj[:num_vis, -1, 0],
        traj[:num_vis, -1, 1],
        c="royalblue",
        s=20,
        label="Generated (t=1)",
    )

    ax.set_title(title, fontsize=28)
    ax.legend()
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def compute_marginal_vector_field(
    schedule, t_val, grid_points, data_samples, device="cpu"
):
    """
    Computes the ideal marginal velocity field v_t(z) at grid points.
    v_t(z) = E[v_t(z|x_1) | z]
           = Integral( v_t(z|x_1) * p(x_1|z) ) dx_1

    Approximated via weighted average over data samples:
    Weights w_i propto p(z | x_1_i) = N(z; alpha*x_1_i, sigma^2 I)
    """
    t_tensor = torch.tensor([t_val], device=device)
    alpha, sigma, d_alpha, d_sigma = schedule.get_coefficients(t_tensor)

    sigma_safe = sigma.clamp(min=1e-3)

    # grid_points: (G, D)
    # data_samples: (N, D)
    G = grid_points.shape[0]
    N = data_samples.shape[0]

    # We want distance between every grid point and every data point scaled by alpha
    # Mean of z given x1 is alpha * x1
    mu = alpha * data_samples  # (N, D)

    grid_exp = grid_points.unsqueeze(1)
    mu_exp = mu.unsqueeze(0)

    # Squared Euclidean distance: ||z - alpha*x1||^2
    dists = torch.sum((grid_exp - mu_exp) ** 2, dim=2)  # (G, N)

    # Log-Weights: -dist / (2 * sigma^2)
    log_weights = -dists / (2 * sigma_safe**2)

    # Softmax over N (data points) to get p(x_1|z)
    weights = torch.softmax(log_weights, dim=1)  # (G, N)

    # 4. Compute Conditional Velocities v(z|x_1)
    # v = d_alpha * x1 + d_sigma * eps
    # eps = (z - alpha * x1) / sigma

    eps_reconstructed = (grid_exp - mu_exp) / sigma_safe

    v_cond = d_alpha * data_samples.unsqueeze(0) + d_sigma * eps_reconstructed

    # 5. Marginal Velocity: Weighted Sum
    # sum over N: weights (G, N, 1) * v_cond (G, N, D) -> (G, D)
    v_marginal = torch.sum(weights.unsqueeze(-1) * v_cond, dim=1)

    return v_marginal


def visualize_path(
    schedule_class, datasets, num_samples=50000, save_path="results/path_evolution.png"
):
    """
    Visualizes the evolution of multiple datasets over time using the Gaussian path.

    Args:
        schedule_class: The class of the schedule to use (e.g., LinearSchedule).
        datasets: A list of tuples, where each tuple is (dataset_object, dataset_name).
                  Example: [(spiral_ds, "Spiral"), (gmm_ds, "GMM")]
        num_samples: Number of samples to use for the density plot.
        save_path: Path to save the resulting image.
    """
    schedule = schedule_class()

    timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_datasets = len(datasets)
    n_rows = n_datasets * 2
    n_cols = len(timesteps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    grid_size = 20
    range_lim = 4.0
    x = np.linspace(-range_lim, range_lim, grid_size)
    y = np.linspace(-range_lim, range_lim, grid_size)
    X, Y = np.meshgrid(x, y)
    grid_flat = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_tensor = torch.from_numpy(grid_flat).float()

    for ds_idx, (dataset, name) in enumerate(datasets):
        row_density = ds_idx * 2
        row_velocity = ds_idx * 2 + 1

        total_len = len(dataset)
        indices = np.random.choice(
            total_len, min(num_samples, total_len), replace=False
        )
        x_1_raw = dataset.data[indices]

        vf_indices = np.random.choice(total_len, min(5000, total_len), replace=False)
        x_1_vf = dataset.data[vf_indices]

        if hasattr(dataset, "P") and dataset.P is not None:
            x_1_2d = x_1_raw @ dataset.P
            x_1_vf_2d = x_1_vf @ dataset.P
        else:
            x_1_2d = x_1_raw
            x_1_vf_2d = x_1_vf

        x_1_2d = torch.from_numpy(x_1_2d).float()
        x_1_vf_2d = torch.from_numpy(x_1_vf_2d).float()

        x_0 = torch.randn_like(x_1_2d)

        for col_idx, t_val in enumerate(timesteps):
            ax_d = axes[row_density, col_idx]

            t_tensor = torch.tensor([t_val])
            alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)

            z_t = alpha * x_1_2d + sigma * x_0
            z_np = z_t.numpy()
            if row_density == 0:
                logging.info(
                    f"Timestep {t_val}, Alpha: {alpha} * x + Sigma: {sigma} * eps"
                )

            ax_d.hist2d(
                z_np[:, 0],
                z_np[:, 1],
                bins=300,
                range=[[-4, 4], [-4, 4]],
                cmap="viridis",
                density=True,
            )

            ax_d.set_xticks([])
            ax_d.set_yticks([])
            ax_d.set_aspect("equal")

            if col_idx == 0:
                ax_d.set_ylabel(name, fontsize=28, fontweight="bold")

            if row_density == 0:
                ax_d.set_title(f"t = {t_val:.2f}", fontsize=28)

            ax_v = axes[row_velocity, col_idx]

            v_field = compute_marginal_vector_field(
                schedule,
                t_val,
                grid_tensor,
                x_1_vf_2d,
            )
            v_np = v_field.numpy()

            norms = np.linalg.norm(v_np, axis=1, keepdims=True)
            v_norm = v_np / (norms + 1e-5)

            ax_v.quiver(
                X,
                Y,
                v_norm[:, 0].reshape(grid_size, grid_size),
                v_norm[:, 1].reshape(grid_size, grid_size),
                scale=25,
                width=0.005,
                color="darkslategray",
            )

            ax_v.set_xlim(-range_lim, range_lim)
            ax_v.set_ylim(-range_lim, range_lim)
            ax_v.set_xticks([])
            ax_v.set_yticks([])
            ax_v.set_aspect("equal")

            if col_idx == 0:
                ax_v.set_ylabel(
                    f"{name}\nVelocity Field", fontsize=28, fontweight="bold"
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    logging.info(f"Saved path evolution visualization to {save_path}")
    plt.close()


def visualize_image_grid(samples, save_path, nrow=8):
    """
    Visualizes a grid of generated images.
    Args:
        samples: numpy array (B, C, H, W) or (B, H, W, C) in range approx [-1, 1]
        save_path: Path to save image
    """
    # Convert to 0-1 range
    samples = np.clip((samples + 1.0) / 2.0, 0.0, 1.0)

    # If channels first (B, C, H, W), transpose to (B, H, W, C) for matplotlib
    if samples.ndim == 4 and samples.shape[1] in [1, 3, 4]:
        samples = np.transpose(samples, (0, 2, 3, 1))

    B, H, W, C = samples.shape

    ncols = int(np.ceil(B / nrow))
    fig, axes = plt.subplots(nrow, ncols, figsize=(ncols * 2, nrow * 2))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < B:
            axes[i].imshow(samples[i])
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_image_trajectory(traj, save_path, num_rows=5):
    """
    Visualizes the evolution of images over time steps.
    Args:
        traj: numpy array (B, Steps, C, H, W)
        save_path: Path to save image
    """
    # traj shape: (B, Steps, C, H, W)
    B, Steps, C, H, W = traj.shape
    num_rows = min(num_rows, B)

    step_indices = np.linspace(0, Steps - 1, 10, dtype=int)

    fig, axes = plt.subplots(
        num_rows, len(step_indices), figsize=(len(step_indices) * 1.5, num_rows * 1.5)
    )

    for row in range(num_rows):
        for col, step_idx in enumerate(step_indices):
            img = traj[row, step_idx]  # (C, H, W)

            # Normalize [-1, 1] -> [0, 1]
            img = np.clip((img + 1.0) / 2.0, 0.0, 1.0)

            # Transpose to (H, W, C)
            img = np.transpose(img, (1, 2, 0))

            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={step_idx / (Steps - 1):.1f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_step_comparison(
    trainers,
    steps_list,
    dataset,
    save_path,
    num_samples=2048,
    row_labels=None,
    perturb_t: float = None,
    perturb_scale: float = 0.0,
    clip_prediction: bool = False,
):
    """
    Creates a grid visualization comparing models (rows) across different
    sampling steps (columns).

    If perturb_t is provided, it overlays the perturbed samples (Red)
    on top of the baseline samples (Blue) to visualize the drift/correction.
    """
    n_rows = len(trainers)
    n_cols = len(steps_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    P = getattr(dataset, "P", None)
    if hasattr(dataset, "data"):
        D = dataset.data.shape[1]
        gt_high = dataset.data[:num_samples]
    else:
        D = 2
        gt_high = np.random.randn(num_samples, 2)

    if P is not None:
        gt_2d = gt_high @ P
    else:
        gt_2d = gt_high

    dataset_name = "unknown"
    if len(trainers) > 0 and hasattr(trainers[0], "config"):
        dataset_name = trainers[0].config.get("dataset_type", "unknown").lower()

    is_imbalanced = any(
        kw in dataset_name for kw in ["imbalanced", "extreme", "long_tail"]
    )

    if is_imbalanced:
        x_min, x_max = gt_2d[:, 0].min(), gt_2d[:, 0].max()
        y_min, y_max = gt_2d[:, 1].min(), gt_2d[:, 1].max()

        x_margin = max((x_max - x_min) * 0.1, 1.0)
        y_margin = max((y_max - y_min) * 0.1, 1.0)

        xlims = (x_min - x_margin, x_max + x_margin)
        ylims = (y_min - y_margin, y_max + y_margin)
    else:
        xlims = (-2.75, 2.75)
        ylims = (-2.75, 2.75)

    for row_idx, trainer in enumerate(trainers):
        sched_type = trainer.config["schedule_type"]
        loss_target = trainer.config.get("loss_target", "")
        model_type = trainer.config.get("model_type", "mlp")
        device = trainer.device

        supports_perturb = True
        diffusion_type = sched_type
        if model_type == "ddgan":
            diffusion_type = "ddgan"
            supports_perturb = False
        elif loss_target == "consistency":
            diffusion_type = "consistency"
            supports_perturb = False

        is_conditional = getattr(trainer, "conditional", False)
        embeddings, attention_mask = None, None
        if is_conditional:
            with torch.no_grad():
                embeddings, attention_mask = trainer.model["text_enc"](
                    [""] * num_samples
                )

        for col_idx, n_steps in enumerate(steps_list):
            ax = axes[row_idx, col_idx]

            extra_kwargs = {}
            if sched_type == "ddpm":
                extra_kwargs["clip_prediction"] = clip_prediction

            # 1. Baseline Sampling (No Perturbation)
            samples_base = generate_samples(
                model=trainer.model,
                schedule=trainer.schedule,
                batch_size=num_samples,
                data_shape=(D,),
                diffusion_type=diffusion_type,
                prediction_target=trainer.prediction_target,
                num_steps=n_steps,
                is_conditional=is_conditional,
                embeddings=embeddings,
                attention_mask=attention_mask,
                projection_matrix=P,
                return_traj=False,
                device=device,
                **extra_kwargs,
            )

            # Plot Baseline (Blue, Background)
            alpha_base = 0.3 if (perturb_t is not None and supports_perturb) else 0.5
            ax.scatter(
                samples_base[:, 0],
                samples_base[:, 1],
                s=1,
                alpha=alpha_base,
                c="royalblue",
                label="Base",
            )

            # 2. Perturbed Sampling
            if perturb_t is not None and supports_perturb:
                samples_perturb = generate_samples(
                    model=trainer.model,
                    schedule=trainer.schedule,
                    batch_size=num_samples,
                    data_shape=(D,),
                    diffusion_type=diffusion_type,
                    prediction_target=trainer.prediction_target,
                    num_steps=n_steps,
                    is_conditional=is_conditional,
                    embeddings=embeddings,
                    attention_mask=attention_mask,
                    projection_matrix=P,
                    return_traj=False,
                    device=device,
                    perturb_t=perturb_t,
                    perturb_scale=perturb_scale,
                    **extra_kwargs,
                )

                # Plot Perturbed (Red, Foreground)
                ax.scatter(
                    samples_perturb[:, 0],
                    samples_perturb[:, 1],
                    s=1,
                    alpha=0.45,
                    c="tomato",
                    label="Perturbed",
                )

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"N = {n_steps}", fontsize=26, fontweight="bold")

            if col_idx == 0:
                if row_labels:
                    label = row_labels[row_idx]
                else:
                    label = f"{sched_type.upper()}"
                ax.set_ylabel(label, fontsize=26, fontweight="bold")

            if (
                row_idx == 0
                and col_idx == 0
                and perturb_t is not None
                and supports_perturb
            ):
                ax.legend(loc="upper right", fontsize=18)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_single_image(image_data, save_path):
    """Helper for thread pool to save a single image."""
    try:
        img = Image.fromarray(image_data)
        img.save(save_path)
    except Exception as e:
        logging.error(f"Failed to save image {save_path}: {e}")


def save_images_parallel(images, save_dir, start_idx=0, num_workers=8):
    """
    Saves a batch of images asynchronously using a thread pool.

    Args:
        images: (B, C, H, W) or (B, H, W, C) array/tensor.
                Range can be [-1, 1] or [0, 1].
        save_dir: Directory to save images.
        start_idx: Starting index for filenames.
        num_workers: Number of threads.
    """
    if isinstance(images, torch.Tensor):
        images = images.float().detach().cpu().numpy()

    if images.min() < 0:
        images = (images + 1.0) / 2.0
    images = np.clip(images, 0.0, 1.0)

    images = (images * 255).astype(np.uint8)

    if images.ndim == 4:
        if images.shape[1] in [1, 3, 4]:
            images = np.transpose(images, (0, 2, 3, 1))

    os.makedirs(save_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, img_data in enumerate(images):
            idx = start_idx + i
            save_path = os.path.join(save_dir, f"{idx:06d}.png")
            if img_data.shape[-1] == 1:
                img_data = img_data.squeeze(-1)
            futures.append(executor.submit(save_single_image, img_data, save_path))

        concurrent.futures.wait(futures)


def visualize_lr_search(lrs, losses, smoothed_losses, suggested_lr, save_path):
    """
    Visualizes the learning rate search results: Loss vs Learning Rate.
    Plots both raw and smoothed loss on log-log and log-linear scales.

    Args:
        lrs: List of learning rates tested.
        losses: List of recorded losses.
        smoothed_losses: List of smoothed losses.
        suggested_lr: The calculated optimal learning rate.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    ax = axes[0]
    ax.plot(lrs, losses, alpha=0.4, label="Raw Loss", color="grey")
    if smoothed_losses is not None:
        ax.plot(lrs, smoothed_losses, "b-", linewidth=2, label="Smoothed Loss")

    if suggested_lr is not None:
        ax.axvline(
            suggested_lr,
            color="green",
            linestyle="--",
            label=f"Suggested LR: {suggested_lr:.2e}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title("LR Range Test (Linear Y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log Y-Axis Plot (Better for seeing orders of magnitude)
    ax = axes[1]
    ax.plot(lrs, losses, alpha=0.4, color="grey")
    if smoothed_losses is not None:
        ax.plot(lrs, smoothed_losses, "b-", linewidth=2)

    if suggested_lr is not None:
        ax.axvline(suggested_lr, color="green", linestyle="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("LR Range Test (Log Y)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Saved LR search plot to {save_path}")


def visualize_cm_evolution(
    model,
    dataset,
    save_path,
    num_samples=2048,
    num_steps=4,
    device="cuda",
    dataset_name="unknown",
):
    """
    Visualizes the evolution of the predicted x0 and xt for different NFE steps.
    Each NFE gets its own column displaying the ground truth, the noisy step (xt),
    and the model's direct prediction (x0).
    """
    gt_high = dataset.data[:num_samples]
    if hasattr(dataset, "P") and dataset.P is not None:
        gt_2d = gt_high @ dataset.P
        D = dataset.P.shape[0]
    else:
        gt_2d = gt_high
        D = gt_high.shape[1]

    is_conditional = (
        isinstance(model, (dict, torch.nn.ModuleDict)) and "text_enc" in model
    )
    embeddings, attention_mask = None, None
    if is_conditional:
        with torch.no_grad():
            embeddings, attention_mask = trainer.model["text_enc"]([""] * num_samples)

    final_samples, (traj_xt, traj_x0, t_steps) = generate_samples(
        model=model,
        schedule=None,
        batch_size=num_samples,
        data_shape=(D,),
        diffusion_type="consistency",
        prediction_target="x",
        num_steps=num_steps,
        is_conditional=is_conditional,
        embeddings=embeddings,
        attention_mask=attention_mask,
        projection_matrix=getattr(dataset, "P", None),
        return_traj=True,
        device=device,
    )

    logging.info(f"\n--- Evaluation Metrics for {save_path} ---")

    # Chamfer Distance (Distribution match, not trajectory)
    chamfer_dist = chamfer_distance(final_samples, gt_2d, is_trajectory=False)
    logging.info(f"  > Distribution Chamfer Distance: {chamfer_dist:.4f}")

    # Precision & Recall (Only for GMM / Pinwheel)
    dataset_name_lower = dataset_name.lower()
    if "gmm" in dataset_name_lower or "pinwheel" in dataset_name_lower:
        k = 8 if "gmm" in dataset_name_lower else 5
        precision, recall = compute_precision_recall(final_samples, gt_2d, k=k)
        logging.info(f"  > Manifold Precision: {precision:.4f}")
        logging.info(f"  > Manifold Recall:    {recall:.4f}")
    else:
        logging.info(f"  > Precision/Recall skipped for dataset: {dataset_name_lower}")

    logging.info("------------------------------------------\n")

    is_imbalanced = any(
        kw in dataset_name_lower for kw in ["imbalanced", "extreme", "long_tail"]
    )

    if is_imbalanced:
        x_min, x_max = gt_2d[:, 0].min(), gt_2d[:, 0].max()
        y_min, y_max = gt_2d[:, 1].min(), gt_2d[:, 1].max()

        x_margin = max((x_max - x_min) * 0.1, 1.0)
        y_margin = max((y_max - y_min) * 0.1, 1.0)

        xlims = (x_min - x_margin, x_max + x_margin)
        ylims = (y_min - y_margin, y_max + y_margin)
    else:
        xlims = (-4, 4)
        ylims = (-4, 4)

    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
    if num_steps == 1:
        axes = [axes]

    for i in range(num_steps):
        ax = axes[i]

        idx = np.random.choice(len(gt_2d), min(2000, len(gt_2d)), replace=False)
        ax.scatter(
            gt_2d[idx, 0],
            gt_2d[idx, 1],
            s=10,
            c="lightgray",
            alpha=0.35 if not is_imbalanced else 0.5,
            label="Data (GT)",
        )

        # Plot xt (Noisy State)
        # we scale it by c_in
        t_curr = t_steps[i]
        sigma_data = 0.5  # Standard for CM/EDM
        c_in = 1.0 / np.sqrt(t_curr**2 + sigma_data**2)

        xt = traj_xt[:, i, :]
        xt_scaled = xt * c_in
        ax.scatter(
            xt_scaled[:, 0],
            xt_scaled[:, 1],
            s=10,
            c="tomato",
            alpha=0.4,
            label="c_{in} * x_t (Scaled Noisy)",
        )

        # Plot x0 (Predicted Data)
        x0 = traj_x0[:, i, :]
        ax.scatter(
            x0[:, 0], x0[:, 1], s=5, c="royalblue", alpha=0.5, label="Predicted x_0"
        )

        ax.set_title(f"Step {i + 1} (NFE={i + 1})", fontsize=28)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.legend(loc="upper right", markerscale=2.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    if dataset_name_lower == "gmm_long_tail":
        save_path_hist = f"{save_path.split('.')[0]}_hist{save_path.split('.')[1]}"
        visualize_long_tail_histogram(final_samples, gt_2d, save_path_hist)


def visualize_long_tail_histogram(
    generated_samples, gt_samples, save_path, title="Long Tail Distribution (X-Axis)"
):
    """
    Custom visualization for the 'gmm_long_tail' dataset.
    Plots the histogram of the X-axis marginal distribution to clearly show
    how well the model captures the exponentially decaying modes.

    Args:
        generated_samples (np.ndarray): Shape (N, D), the predicted/generated points.
        gt_samples (np.ndarray): Shape (M, D), the ground truth data points.
        save_path (str): Path to save the plot.
        title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    gen_x = generated_samples[:, 0]
    gt_x = gt_samples[:, 0]

    ax.hist(
        gt_x,
        bins=150,
        density=True,
        alpha=0.5,
        color="lightgray",
        label="Ground Truth",
        edgecolor="none",
    )

    ax.hist(
        gen_x,
        bins=150,
        density=True,
        alpha=0.6,
        color="royalblue",
        label="Generated",
        edgecolor="none",
    )

    ax.set_title(title, fontsize=28, fontweight="bold")
    ax.set_xlabel("X coordinate", fontsize=28)
    ax.set_ylabel("Density", fontsize=28)

    ax.legend(fontsize=28)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logging.info(f"Saved long tail histogram to {save_path}")
