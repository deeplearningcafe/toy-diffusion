import os
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt

from toy_diffusion.evaluation import compute_fid, compute_curvature
from toy_diffusion.utils.visualization import (
    visualize_image_grid,
    visualize_image_trajectory,
    save_images_parallel,
)


def evaluate_model(
    trainer,
    pred_target,
    save_dir,
    prefix,
    data_shape,
    fid_samples=20000,
    batch_size=128,
    num_steps=50,
    vae_batch_size=32,
    prompts=None,
    filter_type="none",
    keep_ratio=1.0,
    var_timesteps=15,
):
    """
    Evaluates the model by:
    1. Generating a small grid and trajectory visualization.
    2. Generating a large number of samples (fid_samples) to a temp dir.
    3. Computing FID against the training data.
    4. Computing Curvature (Path Straightness).
    """
    logging.info(f"\n--- Evaluating {prefix} ---")

    num_vis = 16
    sample_out = trainer.sample(
        pred_target,
        num_vis,
        data_shape=(*data_shape,),
        num_steps=num_steps,
        vae_batch_size=vae_batch_size,
        prompts=None,
        filter_type=filter_type,
        keep_ratio=keep_ratio,
        var_timesteps=var_timesteps,
    )

    if filter_type != "none":
        final_samples, traj, var_scores = sample_out
    else:
        final_samples, traj = sample_out

    save_path_grid = f"{save_dir}/{prefix}_grid.png"
    visualize_image_grid(final_samples, save_path_grid, nrow=4)
    logging.info(f"Saved grid to {save_path_grid}")

    save_path_traj = f"{save_dir}/{prefix}_traj.png"
    visualize_image_trajectory(traj, save_path_traj, num_rows=4)
    logging.info(f"Saved trajectory to {save_path_traj}")

    curvature = compute_curvature(traj)
    logging.info(f"Path Curvature: {curvature:.4f} (1.0 = Straight)")

    # Create temp directory for generated images
    temp_gen_dir = os.path.join(save_dir, f"temp_{prefix}")
    os.makedirs(temp_gen_dir, exist_ok=True)

    logging.info(f"Generating {fid_samples} samples for FID evaluation...")
    num_batches = (fid_samples + batch_size - 1) // batch_size
    generated_count = 0
    generated_count_total = 0
    all_var_scores = []

    for i in range(num_batches):
        needed = fid_samples - generated_count
        request_bs = min(batch_size, int(np.ceil(needed / keep_ratio)))
        batch_prompts = (
            prompts[generated_count_total : generated_count_total + request_bs]
            if prompts is not None
            else None
        )
        sample_out = trainer.sample(
            pred_target,
            request_bs,
            data_shape=(*data_shape,),
            num_steps=num_steps,
            return_traj=False,
            vae_batch_size=vae_batch_size,
            prompts=batch_prompts,
            filter_type=filter_type,
            keep_ratio=keep_ratio,
            var_timesteps=var_timesteps,
        )

        if filter_type != "none":
            batch_samples, var_scores = sample_out
            all_var_scores.extend(var_scores.tolist())
        else:
            batch_samples = sample_out

        actual_keep = min(len(batch_samples), needed)
        batch_samples = batch_samples[:actual_keep]

        save_images_parallel(
            batch_samples, temp_gen_dir, start_idx=generated_count, num_workers=8
        )
        generated_count += actual_keep
        generated_count_total += request_bs

        if (i + 1) % 10 == 0:
            logging.info(f"Generated {generated_count}/{fid_samples} images...")

    if filter_type != "none" and len(all_var_scores) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(all_var_scores, bins=50, alpha=0.7, color="orange", edgecolor="black")
        plt.title(
            f"Variance Scores Histogram (CFG={trainer.config.get('cfg_scale', 1.0)})"
        )
        plt.xlabel("Variance Score")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.7)
        hist_path = os.path.join(save_dir, f"{prefix}_variance_hist.png")
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Saved variance histogram to {hist_path}")

    real_data_path = trainer.config["data_path"]
    logging.info(f"Computing FID between {real_data_path} and {temp_gen_dir}")

    temp_real_dir = None
    if trainer.dataset.exclude_tags:
        # Create temp directory for real images to compute FID only on the actual training distribution
        temp_real_dir = os.path.join(save_dir, f"temp_real_{prefix}")
        os.makedirs(temp_real_dir, exist_ok=True)

        logging.info(
            f"Copying {len(trainer.dataset.img_paths)} real images to {temp_real_dir} for FID computation..."
        )
        for idx, img_path in enumerate(trainer.dataset.img_paths):
            dest_path = os.path.join(temp_real_dir, f"{idx}_{img_path.name}")
            try:
                shutil.copy(img_path, dest_path)
            except Exception as e:
                logging.error(f"Failed to copy {img_path}: {e}")

        real_data_path = temp_real_dir

    try:
        fid_score = compute_fid(
            real_data_path, temp_gen_dir, batch_size=batch_size, num_workers=4
        )
        logging.info(f"FID Score ({prefix}): {fid_score:.4f}")
    except Exception as e:
        logging.error(f"FID computation failed: {e}")

    shutil.rmtree(temp_gen_dir, ignore_errors=True)
    if temp_real_dir is not None:
        shutil.rmtree(temp_real_dir, ignore_errors=True)
    logging.info("Evaluation complete.\n")
