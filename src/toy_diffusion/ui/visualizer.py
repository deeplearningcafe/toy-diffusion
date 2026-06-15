import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from toy_diffusion.utils.visualization import compute_marginal_vector_field
from toy_diffusion.paths.sampling import sample_euler
from toy_diffusion.paths.scheduler import LinearSchedule, DDPMSchedule
from toy_diffusion.data.synthetic import SyntheticDataset


class Visualizer:
    def __init__(self, device="cpu"):
        self.device = device
        self.VISUAL_OFFSET = 8.0  # Distance between Source and Target centers
        self.LIMIT = 4.0
        self.GRID_RES = 20

    def _get_shift(self, t):
        """
        Calculates the visual shift on the X-axis based on time t.
        t=0 -> Shift = -4 (Left/Source)
        t=1 -> Shift = +4 (Right/Target)
        """
        # Linear interpolation from -VISUAL_OFFSET/2 to +VISUAL_OFFSET/2
        return (t - 0.5) * self.VISUAL_OFFSET

    def _solve_linear_system(
        self, pred, z_t, alpha, sigma, d_alpha, d_sigma, pred_target, view_mode
    ):
        """
        Unified perspective algebraic solver.
        Converts Model Prediction -> Visualization Field (Velocity or Score).
        """
        alpha = alpha.clamp(min=1e-5)
        sigma = sigma.clamp(min=1e-5)

        det = alpha * d_sigma - sigma * d_alpha
        det = torch.where(det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det)

        if pred_target == "v":
            v_pred = pred
            # eps = (alpha * v - d_alpha * z) / det
            eps_pred = (alpha * v_pred - d_alpha * z_t) / det
        elif pred_target == "eps":
            eps_pred = pred
            # x = (z - sigma * eps) / alpha
            x_pred = (z_t - sigma * eps_pred) / alpha
            # v = d_alpha * x + d_sigma * eps
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
        elif pred_target == "x":
            x_pred = pred
            eps_pred = (z_t - alpha * x_pred) / sigma
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
        else:
            raise ValueError(f"Unknown prediction target {pred_target}")

        if view_mode == "Velocity":
            return v_pred
        elif view_mode == "Score":
            # Score \approx -eps / sigma
            return -eps_pred / sigma

        return v_pred

    def _plot_subplot(
        self,
        ax,
        session,
        t_val,
        view_mode,
        is_ground_truth,
        shift_x,
        grid_tensor,
        X_g,
        Y_g,
    ):
        """
        Helper to render a single subplot (GT or Model).
        """
        ax.set_facecolor("black")
        ax.set_xlim(
            -self.LIMIT - self.VISUAL_OFFSET / 2, self.LIMIT + self.VISUAL_OFFSET / 2
        )
        ax.set_ylim(-self.LIMIT, self.LIMIT)
        ax.axis("off")

        noise_center = -self.VISUAL_OFFSET / 2
        data_center = self.VISUAL_OFFSET / 2

        ax.add_patch(plt.Circle((noise_center, 0), 3, color="tomato", alpha=0.1))

        if session.dataset is not None:
            data = session.dataset.data[:512]
            ax.scatter(
                data[:, 0] + data_center, data[:, 1], s=10, c="royalblue", alpha=0.15
            )

        title = (
            "Ground Truth (Analytic)"
            if is_ground_truth
            else "Model Prediction (Learned)"
        )
        ax.text(
            0,
            self.LIMIT - 0.5,
            title,
            color="white",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

        # 2. Current State Particles (z_t)
        # For comparison, we use the analytic GT path for both plots to show
        t_tensor = torch.tensor([t_val], device=self.device)
        alpha, sigma, d_alpha, d_sigma = session.schedule.get_coefficients(t_tensor)

        data_tensor = torch.from_numpy(data).to(self.device)
        noise = torch.randn_like(data_tensor)

        x_t = alpha * data_tensor + sigma * noise
        x_np = x_t.cpu().numpy()

        ax.scatter(
            x_np[:, 0] + shift_x, x_np[:, 1], s=10, c="white", alpha=0.4, zorder=3
        )

        if session.vis_noise is not None and session.vis_data is not None:
            z_noise = session.vis_noise.to(self.device)
            x_data = session.vis_data.to(self.device)

            z_t = alpha * x_data + sigma * z_noise
            z_np = z_t.cpu().numpy()

            ax.scatter(
                z_np[:, 0] + shift_x, z_np[:, 1], s=20, c="tomato", alpha=0.9, zorder=5
            )

        if view_mode == "Trajectory":
            if is_ground_truth:
                s_steps = torch.linspace(0, 1, 50, device=self.device)
                alphas, sigmas, _, _ = session.schedule.get_coefficients(s_steps)

                alphas = alphas.view(-1, 1, 1)
                sigmas = sigmas.view(-1, 1, 1)

                z_path = alphas * x_data.unsqueeze(0) + sigmas * z_noise.unsqueeze(0)
                z_path_np = z_path.cpu().numpy()

                for i in range(len(z_noise)):
                    path_x = (
                        z_path_np[:, i, 0]
                        + (s_steps.cpu().numpy() - 0.5) * self.VISUAL_OFFSET
                    )
                    path_y = z_path_np[:, i, 1]
                    ax.plot(path_x, path_y, c="limegreen", alpha=0.4, linewidth=1)

            else:
                if session.model_traj_cache is None:
                    with torch.no_grad():
                        _, traj = sample_euler(
                            session.model,
                            session.schedule,
                            session.prediction_target,
                            num_steps=50,
                            batch_size=len(z_noise),
                            data_shape=(2,),
                            noise=z_noise,
                            device=self.device,
                            return_traj=True,
                        )
                    session.model_traj_cache = traj  # (B, Steps, 2)

                traj = session.model_traj_cache
                s_steps = np.linspace(0, 1, traj.shape[1])

                for i in range(len(z_noise)):
                    # Shift X
                    path_x = traj[i, :, 0] + (s_steps - 0.5) * self.VISUAL_OFFSET
                    path_y = traj[i, :, 1]
                    ax.plot(path_x, path_y, c="orchid", alpha=0.4, linewidth=1)

        else:
            grid_model = grid_tensor.clone()
            grid_model[:, 0] -= shift_x  # Shift back to model space

            if is_ground_truth:
                # GT Marginal Field
                subset_data = (
                    torch.from_numpy(session.dataset.data[:512]).to(self.device).float()
                )
                v_field = compute_marginal_vector_field(
                    session.schedule, t_val, grid_model, subset_data, device=self.device
                )

                field = v_field
                color = "limegreen"
            else:
                # Model Field
                with torch.no_grad():
                    t_batch = t_tensor.repeat(len(grid_model))
                    raw_pred = session.model(grid_model, t_batch)

                    field = self._solve_linear_system(
                        raw_pred,
                        grid_model,
                        alpha,
                        sigma,
                        d_alpha,
                        d_sigma,
                        session.prediction_target,
                        view_mode,
                    )
                color = "tomato" if view_mode == "Velocity" else "orchid"

            f_np = field.cpu().numpy()
            norm = np.linalg.norm(f_np, axis=1, keepdims=True) + 1e-5
            f_norm = f_np / norm

            ax.quiver(
                X_g,
                Y_g,
                f_norm[:, 0],
                f_norm[:, 1],
                color=color,
                alpha=0.6,
                scale=30,
                width=0.004,
            )

    def render_exploration_plot(self, session, t_val, view_mode, show_gt, show_model):
        """
        Renders the Tab 1 exploration plot with visual shifting.
        Supports 2 rows if model comparison is active.
        """
        has_model = show_model and (session.model is not None)
        rows = 2 if has_model else 1

        fig = plt.figure(figsize=(10, 5 * rows), dpi=100, facecolor="black")

        shift_x = self._get_shift(t_val)

        x_g = np.linspace(-self.LIMIT + shift_x, self.LIMIT + shift_x, self.GRID_RES)
        y_g = np.linspace(-self.LIMIT, self.LIMIT, self.GRID_RES)
        X_g, Y_g = np.meshgrid(x_g, y_g)
        grid_flat_vis = np.stack([X_g.flatten(), Y_g.flatten()], axis=1)
        grid_tensor = torch.from_numpy(grid_flat_vis).float().to(self.device)

        ax1 = fig.add_subplot(rows, 1, 1)
        if show_gt:
            self._plot_subplot(
                ax1,
                session,
                t_val,
                view_mode,
                is_ground_truth=True,
                shift_x=shift_x,
                grid_tensor=grid_tensor,
                X_g=X_g,
                Y_g=Y_g,
            )

        if has_model:
            ax2 = fig.add_subplot(rows, 1, 2)
            self._plot_subplot(
                ax2,
                session,
                t_val,
                view_mode,
                is_ground_truth=False,
                shift_x=shift_x,
                grid_tensor=grid_tensor,
                X_g=X_g,
                Y_g=Y_g,
            )

        axes = [ax1]
        if has_model:
            axes.append(ax2)

        for ax in axes:
            for traj_data in session.user_trajectories:
                full_traj = traj_data["traj"]
                t_start = traj_data["t_start"]

                if t_val < t_start:
                    continue

                duration = 1.0 - t_start
                if duration <= 1e-5:
                    continue

                progress = (t_val - t_start) / duration
                idx = int(progress * len(full_traj))
                idx = np.clip(idx, 1, len(full_traj))

                path_segment = full_traj[:idx]
                steps_total = len(full_traj)
                t_points = np.linspace(
                    t_start, t_start + (idx / steps_total) * (1 - t_start), idx
                )
                shift_points = (t_points - 0.5) * self.VISUAL_OFFSET

                path_vis_x = path_segment[:, 0] + shift_points
                path_vis_y = path_segment[:, 1]

                ax.plot(path_vis_x, path_vis_y, c="white", linewidth=2, alpha=0.8)
                ax.scatter(path_vis_x[-1], path_vis_y[-1], c="white", s=20)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0.1, facecolor="black"
        )
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def integrate_particle(
        self, session, visual_start_x, visual_start_y, t_start, perturb_scale=0.0
    ):
        """
        Integrates a particle starting from visual coordinates at t_start.
        Returns trajectory in MODEL space.
        """
        shift_start = self._get_shift(t_start)
        model_start_x = visual_start_x - shift_start
        model_start_y = visual_start_y

        z = torch.tensor([[model_start_x, model_start_y]], device=self.device)

        steps = 50
        dt = (1.0 - t_start) / steps
        traj = [z.cpu().numpy()[0]]

        curr_t = t_start

        for _ in range(steps):
            t_tensor = torch.tensor([curr_t], device=self.device)
            alpha, sigma, d_alpha, d_sigma = session.schedule.get_coefficients(t_tensor)

            if session.model is not None:
                with torch.no_grad():
                    raw_pred = session.model(z, t_tensor)
                    v = self._solve_linear_system(
                        raw_pred,
                        z,
                        alpha,
                        sigma,
                        d_alpha,
                        d_sigma,
                        session.prediction_target,
                        "Velocity",
                    )
            else:
                v = torch.zeros_like(z)

            noise = torch.randn_like(z) * perturb_scale

            z = z + v * dt + noise * np.sqrt(dt)
            curr_t += dt
            traj.append(z.cpu().numpy()[0])

        return np.array(traj)

    def render_sampling_animation(self, samples_list):
        """
        Converts a list of sample arrays (from the sampling loop) into a list of PIL images.
        Used for the 'Sampling' tab animation.
        """
        images = []
        for i, samples in enumerate(samples_list):
            fig = plt.figure(figsize=(4, 4), dpi=80, facecolor="black")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_facecolor("black")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis("off")

            # samples is np.array
            ax.scatter(samples[:, 0], samples[:, 1], s=2, c="white", alpha=0.7)

            ax.text(-3.5, 3.5, f"Step {i}", color="white")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            images.append(Image.open(buf))
        return images

    def render_forward_diffusion(self, image_pil, t_val, schedule_type):
        """Renders a single frame of the forward diffusion process (adding noise) to an image."""
        if image_pil is None:
            return None

        img_np = np.array(image_pil.convert("RGB")) / 255.0
        img_tensor = (
            torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        )
        img_tensor = img_tensor.to(self.device)

        if schedule_type == "DDPM":
            schedule = DDPMSchedule(device=self.device)
        else:
            schedule = LinearSchedule(device=self.device)

        # Invert time: t_val=0 -> Clean Data, t_val=1 -> Pure Noise
        t_tensor = torch.tensor([1.0 - t_val], device=self.device)
        alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)

        generator = torch.Generator(device=self.device).manual_seed(42)
        noise = torch.randn(img_tensor.shape, generator=generator, device=self.device)

        z_t = alpha.view(-1, 1, 1, 1) * img_tensor + sigma.view(-1, 1, 1, 1) * noise

        z_t_np = z_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        z_t_np = np.clip((z_t_np + 1.0) / 2.0, 0.0, 1.0)

        return Image.fromarray((z_t_np * 255).astype(np.uint8))

    def render_forward_diffusion_plot(self, image_pil, schedule_type):
        """Generates a static plot of the forward diffusion process across 5 timesteps."""
        if image_pil is None:
            return None

        orig_W, orig_H = image_pil.size

        img_np = np.array(image_pil.convert("RGB")) / 255.0
        img_tensor = (
            torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
        )
        img_tensor = img_tensor.to(self.device)

        if schedule_type == "DDPM":
            schedule = DDPMSchedule(device=self.device)
        else:
            schedule = LinearSchedule(device=self.device)

        timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]

        base_height = 8.0
        aspect_ratio = orig_W / orig_H
        single_width = base_height * aspect_ratio
        total_width = 5 * single_width

        fig, axes = plt.subplots(1, 5, figsize=(total_width, base_height + 0.6))
        fig.patch.set_facecolor("black")

        generator = torch.Generator(device=self.device).manual_seed(42)
        noise = torch.randn(img_tensor.shape, generator=generator, device=self.device)

        for i, t_val in enumerate(timesteps):
            t_tensor = torch.tensor([1.0 - t_val], device=self.device)
            alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)

            z_t = alpha.view(-1, 1, 1, 1) * img_tensor + sigma.view(-1, 1, 1, 1) * noise
            z_t_np = z_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            z_t_np = np.clip((z_t_np + 1.0) / 2.0, 0.0, 1.0)

            ax = axes[i]
            ax.imshow(z_t_np)
            ax.axis("off")
            ax.set_title(f"t={t_val}", color="white", fontsize=24, pad=15)

        plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99, bottom=0.01, top=0.90)

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            facecolor="black",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def render_image_patches(self, image_pil, patch_size):
        """Visualizes DiT image patching using an exploded grid view."""
        if image_pil is None:
            return None

        img_np = np.array(image_pil.convert("RGB"))
        H, W, C = img_np.shape

        new_H = (H // patch_size) * patch_size
        new_W = (W // patch_size) * patch_size

        if new_H == 0 or new_W == 0:
            return image_pil

        img_np = img_np[:new_H, :new_W, :]

        gap = max(2, int(patch_size * 0.15))
        grid_H = new_H // patch_size
        grid_W = new_W // patch_size

        out_H = grid_H * patch_size + (grid_H - 1) * gap
        out_W = grid_W * patch_size + (grid_W - 1) * gap

        # Use a dark gray background for the gaps to make the patches pop
        out_img = np.ones((out_H, out_W, C), dtype=np.uint8) * 30

        for i in range(grid_H):
            for j in range(grid_W):
                patch = img_np[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                    :,
                ]
                y_start = i * (patch_size + gap)
                x_start = j * (patch_size + gap)
                out_img[
                    y_start : y_start + patch_size, x_start : x_start + patch_size, :
                ] = patch

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")

        ax.imshow(out_img)
        ax.axis("off")

        ax.set_title(
            f"DiT Patching Visualization\nPatch Size: {patch_size}x{patch_size} | Total Patches: {grid_H * grid_W}",
            color="white",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(
            buf, format="png", facecolor="#1e1e1e", bbox_inches="tight", dpi=300
        )
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def render_ground_truth(self, dataset_name="spiral", num_samples=1000):
        """
        Generates a plot comparing the forward paths of Linear and DDPM schedules.
        """
        dataset = SyntheticDataset(name=dataset_name, n_samples=num_samples)
        schedules = [
            ("Linear (Flow Matching)", LinearSchedule(self.device)),
            ("DDPM (VP SDE)", DDPMSchedule(self.device)),
        ]

        # Colors
        c_data = "royalblue"
        c_mid = "gold"
        c_noise = "tomato"
        c_bg = "lightgray"

        gt_high = dataset.data[:num_samples]
        P = getattr(dataset, "P", None)
        if P is not None:
            gt_2d = gt_high @ P
        else:
            gt_2d = gt_high

        is_imbalanced = any(
            kw in dataset_name for kw in ["imbalanced", "extreme", "long_tail"]
        )

        if is_imbalanced:
            x_min, x_max = gt_2d[:, 0].min(), gt_2d[:, 0].max()
            y_min, y_max = gt_2d[:, 1].min(), gt_2d[:, 1].max()

            x_margin = max((x_max - x_min) * 0.1, 1.0)
            y_margin = max((y_max - y_min) * 0.1, 1.0)

            x_bounds = [x_min - x_margin, x_max + x_margin]
            y_bounds = [y_min - y_margin, y_max + y_margin]
        else:
            scale = 4.0
            x_bounds = [-scale, scale]
            y_bounds = [-scale, scale]

        z_noise = torch.randn(num_samples, gt_high.shape[1], device=self.device)
        gt_tensor = torch.from_numpy(gt_high).float().to(self.device)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(
            "Ground-Truth Conditional Probability Path", fontsize=26, fontweight="bold"
        )
        fig.patch.set_facecolor("white")

        legend_size = 22
        markerscale = 2.0

        for col_idx, (name, schedule) in enumerate(schedules):
            ax = axes[col_idx]
            ax.set_title(name, fontsize=24)

            ax.scatter(gt_2d[:, 0], gt_2d[:, 1], s=10, c=c_bg, alpha=0.35, label="Data")
            ax.set_xlim(*x_bounds)
            ax.set_ylim(*y_bounds)
            ax.set_xticks([])
            ax.set_yticks([])

            ts_vis = torch.linspace(0, 1, 20).to(self.device)
            alpha_vis, sigma_vis, _, _ = schedule.get_coefficients(ts_vis)
            alpha_vis = alpha_vis.view(-1, 1)
            sigma_vis = sigma_vis.view(-1, 1)

            for i in range(min(20, num_samples)):
                path = alpha_vis * gt_tensor[i].unsqueeze(0) + sigma_vis * z_noise[
                    i
                ].unsqueeze(0)

                path_2d = path.cpu().numpy()
                if P is not None:
                    path_2d = path_2d @ P

                ax.plot(
                    path_2d[:, 0], path_2d[:, 1], c="black", alpha=0.8, linewidth=1.5
                )

            time_steps = [0.0, 0.5, 1.0]
            time_colors = [c_noise, c_mid, c_data]

            for t_val, color in zip(time_steps, time_colors):
                t_tensor = torch.tensor([t_val], device=self.device)
                alpha, sigma, _, _ = schedule.get_coefficients(t_tensor)

                z_t = alpha * gt_tensor + sigma * z_noise
                z_t_2d = z_t.cpu().numpy()
                if P is not None:
                    z_t_2d = z_t_2d @ P

                ax.scatter(
                    z_t_2d[:, 0],
                    z_t_2d[:, 1],
                    s=15,
                    alpha=0.6,
                    c=color,
                    label=f"t={t_val}",
                )

            ax.legend(
                prop={"size": legend_size}, loc="upper right", markerscale=markerscale
            )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor="white", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
