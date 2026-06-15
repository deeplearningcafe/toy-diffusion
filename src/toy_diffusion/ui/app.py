import sys
import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from toy_diffusion.data.synthetic import SyntheticDataset
from toy_diffusion.paths.scheduler import LinearSchedule, DDPMSchedule, VESchedule
from toy_diffusion.trainer import Trainer
from toy_diffusion.paths.sampling import sample_euler, sample_ddim
from toy_diffusion.ui.visualizer import Visualizer


class SessionState:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = config["dataset_type"]
        self.dataset = None
        self.model = None
        self.schedule = LinearSchedule(device=self.device)
        self.prediction_target = "v"
        self.user_trajectories = []

        self.vis_noise = torch.randn(20, 2)
        self.vis_data = None
        self.model_traj_cache = None

        self.load_dataset("spiral")

    def load_dataset(self, name):
        self.dataset_type = name
        self.dataset = SyntheticDataset(
            name=name,
            n_samples=self.config["n_samples"],
            projection_dim=0,
            font_path=self.config["font_path"],
        )
        self.user_trajectories = []
        self.model_traj_cache = None

        if self.dataset is not None and len(self.dataset) > 0:
            idxs = np.random.choice(len(self.dataset), 20)
            self.vis_data = torch.from_numpy(self.dataset.data[idxs]).float()


vis = Visualizer(device="cuda" if torch.cuda.is_available() else "cpu")


def handle_dataset_change(state, name):
    state.load_dataset(name)
    return state


def handle_schedule_change(state, name):
    if name == "Linear (Flow Matching)":
        state.schedule = LinearSchedule(device=state.device)
    elif name == "DDPM":
        state.schedule = DDPMSchedule(device=state.device)
    elif name == "Karras VE":
        state.schedule = VESchedule(device=state.device)
    return state


def handle_click(state, evt: gr.SelectData, t_val, perturb_scale):
    if state.model is None:
        return state

    rows = 2 if (state.model is not None) else 1
    total_h = 500 * rows

    x_px, y_px = evt.index[0], evt.index[1]

    plot_h_px = total_h / rows

    rel_y_px = y_px % plot_h_px

    vis_w = vis.LIMIT * 2 + vis.VISUAL_OFFSET
    vis_h = vis.LIMIT * 2

    x_vis = -(vis_w / 2) + (x_px / 1000) * vis_w

    x_vis = -(vis_w / 2) + (x_px / 1000) * vis_w
    y_vis = (vis_h / 2) - (rel_y_px / 500) * vis_h

    traj = vis.integrate_particle(state, x_vis, y_vis, t_val, perturb_scale)
    state.user_trajectories.append({"traj": traj, "t_start": t_val})
    return state


def run_training(state, pred_target, loss_target, epochs):
    state.prediction_target = pred_target

    trainer = Trainer(state.config, state.dataset, prediction_target=pred_target)
    state.model = trainer.model

    dataloader = torch.utils.data.DataLoader(
        state.dataset, batch_size=256, shuffle=True
    )

    log = f"Started Training {pred_target}-pred...\n"
    yield log, state

    for epoch in range(1, int(epochs) + 1):
        loss = trainer.train_epoch(dataloader)
        if epoch % 1 == 0 or epoch == 1:
            log += f"Ep {epoch}: Loss {loss:.5f}\n"
            yield log, state

    log += "Done!"
    yield log, state


def run_sampling_animation(state, sampler_name, steps, perturb_t, perturb_scale):
    if state.model is None:
        return []

    B = 500
    D = 2

    if "Euler" in sampler_name:
        _, traj = sample_euler(
            state.model,
            state.schedule,
            state.prediction_target,
            num_steps=steps,
            batch_size=B,
            data_shape=(D,),
            device=state.device,
            return_traj=True,
            perturb_t=perturb_t,
            perturb_scale=perturb_scale,
        )
    else:
        _, traj = sample_ddim(
            state.model,
            state.schedule,
            state.prediction_target,
            num_steps=steps,
            batch_size=B,
            data_shape=(D,),
            device=state.device,
            return_traj=True,
        )

    # traj is (B, Steps, D) -> list of np.arrays (Steps, B, D)
    steps_data = [traj[:, i, :] for i in range(traj.shape[1])]

    images = vis.render_sampling_animation(steps_data)
    return images


def create_ui(args):
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)
    print(cfg)

    print("--- UI ---")
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    device = cfg.training.device if torch.cuda.is_available() else "cpu"

    config = {
        **OmegaConf.to_container(cfg.experiment),
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.training),
        **OmegaConf.to_container(cfg.diffusion),
        **OmegaConf.to_container(cfg.model),
        "device": device,
    }

    config.setdefault("perturb_t", 0.5)
    config.setdefault("perturb_scale", 0.4)

    D = 2
    config["projection_dim"] = D
    batch_size = config["batch_size"]

    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#1e1e1e",
        block_background_fill="#2d2d2d",
        body_text_color="white",
        block_label_text_color="white",
        block_title_text_color="white",
    )

    js_func = "document.body.classList.toggle('dark', true);"

    with gr.Blocks(title="Diffusion Explorer", theme=theme, js=js_func) as demo:
        state = gr.State(SessionState(config))

        gr.Markdown("# 🌌 Diffusion Explorer: Flow Matching vs DDPM")

        with gr.Row():
            ds_dd = gr.Dropdown(
                ["spiral", "gmm", "pinwheel", "kanji"], label="Dataset", value="spiral"
            )
            sched_dd = gr.Dropdown(
                ["Linear (Flow Matching)", "DDPM"],
                label="Schedule",
                value="Linear (Flow Matching)",
            )

        with gr.Tabs():
            # TAB 1: Exploration

            with gr.TabItem("📈 Exploration & Physics"):
                with gr.Row():
                    with gr.Column(scale=3):
                        plot_out = gr.Image(
                            label="Distribution Flow (Click to Spawn)",
                            height=800,
                            interactive=False,
                        )
                        t_slider = gr.Slider(
                            0.0, 1.0, value=0.0, label="Time t (Source -> Target)"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Visualization Settings")
                        view_mode = gr.Radio(
                            ["Field", "Trajectory", "Score"],
                            label="Visualization Mode",
                            value="Field",
                        )

                        show_gt = gr.Checkbox(label="Show Ground Truth", value=True)
                        show_model = gr.Checkbox(
                            label="Show Model Prediction", value=False
                        )

                        gr.Markdown("### Interaction")
                        perturb_sl = gr.Slider(
                            0.0, 1.0, value=0.0, label="Click Perturbation"
                        )
                        clear_btn = gr.Button("Clear Particles")

            # TAB 2: Training & Sampling
            with gr.TabItem("🛠️ Training & Sampling"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Configuration")
                        pred_dd = gr.Dropdown(
                            ["v", "eps", "x"], label="Prediction Target", value="v"
                        )
                        loss_dd = gr.Dropdown(
                            ["v", "eps", "x"], label="Loss Target", value="v"
                        )
                        epochs_sl = gr.Slider(1, 20, value=10, step=1, label="Epochs")
                        train_btn = gr.Button("Train Model", variant="primary")
                        train_log = gr.Textbox(label="Logs", lines=10)

                    with gr.Column():
                        gr.Markdown("### Sampling")
                        sampler_dd = gr.Dropdown(
                            ["Euler (ODE)", "DDIM (ODE)"],
                            label="Sampler",
                            value="Euler (ODE)",
                        )
                        sample_steps = gr.Slider(
                            10, 100, value=50, step=1, label="Steps"
                        )

                        gr.Markdown("#### SDE / Perturbation")
                        sde_t = gr.Slider(
                            0.0, 1.0, value=0.5, label="Perturbation Time"
                        )
                        sde_scale = gr.Slider(
                            0.0, 1.0, value=0.0, label="Perturbation Scale"
                        )

                        sample_btn = gr.Button(
                            "Run Sampling Animation", variant="primary"
                        )
                        gallery = gr.Gallery(
                            label="Sampling Process", columns=5, height=300
                        )

            # TAB 3: Forward Process
            with gr.TabItem("🔄 Forward Process"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Image Forward Diffusion")
                        forward_input_img = gr.Image(label="Input Image", type="pil")
                        forward_schedule_dd = gr.Dropdown(
                            ["Linear (Flow Matching)", "DDPM"],
                            label="Schedule",
                            value="Linear (Flow Matching)",
                        )
                        forward_t_slider = gr.Slider(
                            0.0, 1.0, value=0.0, label="Time t (Clean -> Noise)"
                        )
                        forward_plot_btn = gr.Button("Generate Static Plot")

                        gr.Markdown("### Thesis Ground Truth Visualizations")
                        thesis_gt_btn = gr.Button("Generate Thesis Ground Truth Paths")
                    with gr.Column():
                        forward_output_img = gr.Image(
                            label="Noisy Image", interactive=False
                        )
                        forward_plot_out = gr.Image(
                            label="Static Plot", interactive=False
                        )
                        thesis_gt_out = gr.Image(
                            label="Ground Truth Paths Comparison", interactive=False
                        )

            # TAB 4: DiT Patching
            with gr.TabItem("🧩 DiT Patching"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Diffusion Transformer Image Patching")
                        patch_input_img = gr.Image(label="Input Image", type="pil")
                        patch_size_slider = gr.Slider(
                            4, 64, value=16, step=4, label="Patch Size"
                        )
                    with gr.Column():
                        patch_output_img = gr.Image(
                            label="Patched Image", interactive=False
                        )

        def update_plot(st, t, vm, gt, md):
            return vis.render_exploration_plot(st, t, vm, gt, md)

        plot_inputs = [state, t_slider, view_mode, show_gt, show_model]

        ds_dd.change(handle_dataset_change, [state, ds_dd], state).then(
            update_plot, plot_inputs, plot_out
        )
        sched_dd.change(handle_schedule_change, [state, sched_dd], state).then(
            update_plot, plot_inputs, plot_out
        )

        # Visualization Updates
        t_slider.change(update_plot, plot_inputs, plot_out)
        view_mode.change(update_plot, plot_inputs, plot_out)
        show_gt.change(update_plot, plot_inputs, plot_out)
        show_model.change(update_plot, plot_inputs, plot_out)

        plot_out.select(handle_click, [state, t_slider, perturb_sl], state).then(
            update_plot, plot_inputs, plot_out
        )
        clear_btn.click(
            lambda s: (s.user_trajectories.clear(), s)[1], state, state
        ).then(update_plot, plot_inputs, plot_out)

        train_btn.click(
            run_training, [state, pred_dd, loss_dd, epochs_sl], [train_log, state]
        ).then(
            lambda: True,
            None,
            show_model,
        ).then(update_plot, plot_inputs, plot_out)

        sample_btn.click(
            run_sampling_animation,
            [state, sampler_dd, sample_steps, sde_t, sde_scale],
            gallery,
        )

        def update_forward_img(img, t, sched):
            if img is None:
                return None
            return vis.render_forward_diffusion(img, t, sched)

        forward_inputs = [forward_input_img, forward_t_slider, forward_schedule_dd]
        forward_input_img.change(update_forward_img, forward_inputs, forward_output_img)
        forward_t_slider.change(update_forward_img, forward_inputs, forward_output_img)
        forward_schedule_dd.change(
            update_forward_img, forward_inputs, forward_output_img
        )

        def generate_forward_plot(img, sched):
            if img is None:
                return None
            return vis.render_forward_diffusion_plot(img, sched)

        forward_plot_btn.click(
            generate_forward_plot,
            [forward_input_img, forward_schedule_dd],
            forward_plot_out,
        )

        def generate_thesis_gt(dataset_name):
            return vis.render_ground_truth(dataset_name)

        thesis_gt_btn.click(
            generate_thesis_gt,
            inputs=[ds_dd],
            outputs=[thesis_gt_out],
        )

        def update_patch_img(img, p_size):
            if img is None:
                return None
            return vis.render_image_patches(img, p_size)

        patch_input_img.change(
            update_patch_img, [patch_input_img, patch_size_slider], patch_output_img
        )
        patch_size_slider.change(
            update_patch_img, [patch_input_img, patch_size_slider], patch_output_img
        )
        demo.load(update_plot, plot_inputs, plot_out)

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
