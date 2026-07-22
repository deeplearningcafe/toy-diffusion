import torch
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import logging
import toml
import os
import random

from toy_diffusion.utils.trainer_utils import (
    get_model,
    get_schedule_loss,
    create_optim_scheduler,
    EMAModel,
    save_checkpoint,
    load_from_checkpoint,
    gpu_setup,
    get_vae,
)
from toy_diffusion.paths.sampling import generate_samples
from toy_diffusion.utils.visualization import (
    visualize_image_grid,
    visualize_image_trajectory,
)

try:
    from toy_diffusion.utils.act_grad_checkpointing import (
        patch_unsloth_smart_gradient_checkpointing,
        patch_torch_compile,
        patch_compiled_autograd,
        CPUGradientAccumulator,
    )
except Exception as e:
    logging.info(f"Can't use unsloth gradient checkpoint {e}")


class Trainer:
    def __init__(
        self,
        config,
        dataset,
        prediction_target,
        pretrained_model=None,
        autocast_enabled=True,
    ):
        """
        Modified to accept a pretrained_model.
        If pretrained_model is provided, we reuse it instead of initializing a new one.
        """
        self.config = config
        self.dataset = dataset
        self.conditional = config.get("is_conditional", False)
        self.device = config["device"]
        self.prediction_target = prediction_target
        self.autocast_enabled = autocast_enabled
        self.autocast_dtype = (
            gpu_setup(self.device) if self.autocast_enabled else torch.float32
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.autocast_dtype == torch.float16
            else None
        )
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self.is_latents = self.config.get("is_latents", False)
        self.gradient_accumulation_steps = self.config.get(
            "gradient_accumulation_steps", 1
        )

        self.vae = None
        if self.is_latents and self.vae is None:
            self.vae = get_vae(self.config, self.device, self.autocast_dtype)

        # Inject vocab properties into the config before building the model
        if self.conditional and hasattr(self.dataset, "vocab"):
            self.config["vocab"] = self.dataset.vocab
            self.config["max_seq_len"] = self.dataset.tiers_len[-1]
            self.config["tiers_len"] = self.dataset.tiers_len
            if self.config.get("cross_attention_dim") is None:
                self.config["cross_attention_dim"] = 256

        if pretrained_model is not None:
            logging.info("Loading pretrained model...")
            self.model = pretrained_model
            self.model.to(self.device)
        else:
            self.model = get_model(config, self.device)
        self.schedule, self.loss_fn, self.model = get_schedule_loss(
            config, self.model, prediction_target, self.device
        )

        self.optimizer, self.scheduler = create_optim_scheduler(
            self.model,
            len_train_loader=len(dataset) / config.get("batch_size", 64),
            conf=config,
        )
        self.grad_offloader = None
        if self.gradient_accumulation_steps > 1:
            target_model = (
                self.model["unet"]
                if (
                    isinstance(self.model, torch.nn.ModuleDict) and "unet" in self.model
                )
                else self.model
            )
            self.grad_offloader = CPUGradientAccumulator(target_model)

        self.use_ema_config = config.get("use_ema", True)
        self.ema_start_epoch = config.get("ema_start_epoch", 0)
        ema_decay = config.get("ema_decay", 0.9999)

        # Initialize EMAModel without creating weights if we haven't reached start epoch
        init_ema_now = self.use_ema_config and (self.ema_start_epoch == 0)
        self.ema = EMAModel(
            model=self.model if init_ema_now else None,
            decay=ema_decay,
            use_ema=init_ema_now,
        )

        self.sample_configs = self._load_sample_configs()

        self.start_epoch = 0
        resume_dir = config.get("resume_from_checkpoint", None)
        if resume_dir is not None:
            self.start_epoch = load_from_checkpoint(
                checkpoint_dir=resume_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                ema=self.ema,
            )
            if self.use_ema_config and self.start_epoch >= self.ema_start_epoch:
                if self.ema.ema_model is None:
                    self.ema.initialize(self.model)

        if config.get("use_gradient_checkpointing", False):
            patch_unsloth_smart_gradient_checkpointing(dtype=torch.bfloat16)

        if hasattr(torch, "compile") and config.get("compile_model", True):
            logging.info("Compiling model with torch.compile for faster training...")
            patch_torch_compile()
            patch_compiled_autograd()
            if isinstance(self.model, torch.nn.ModuleDict) and "unet" in self.model:
                self.model["unet"] = torch.compile(self.model["unet"])
                # skip text encoder compile
            else:
                self.model = torch.compile(self.model)

            if self.vae is not None:
                self.vae = torch.compile(self.vae)

    def _load_sample_configs(self):
        configs = []
        config_file = self.config.get("sample_file", None)
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                toml_conf = toml.load(f)
                defaults = toml_conf.get("prompt", {})
                subsets = defaults.pop("subset", [])
                for sub in subsets:
                    c = defaults.copy()
                    c.update(sub)
                    configs.append(c)
        print(f"Loaded {len(configs)} prompts.")
        return configs

    def train_step(self, batch):
        prompt = None
        if self.conditional:
            x, prompt_tokens, prompt_mask = batch
            prompt = (prompt_tokens, prompt_mask)
        else:
            x = batch
        # Handle Tuple Batches (Reflow)
        if isinstance(x, (list, tuple)):
            x = [b.to(self.device).float() for b in x]
        else:
            x = x.to(self.device).float()

        with torch.autocast(
            device_type=self.device,
            dtype=self.autocast_dtype,
            enabled=self.autocast_enabled,
        ):
            if self.config.get("model_type") == "ddgan":
                loss_d, loss_g = self.loss_fn(self.model, x, prompt)
            else:
                loss = self.loss_fn(self.model, x, prompt)

        if self.config.get("model_type") == "ddgan":
            self.optimizer["D"].zero_grad()
            self.optimizer["G"].zero_grad()

            loss_g.backward(retain_graph=True)
            self.optimizer["G"].step()

            self.optimizer["D"].zero_grad()

            loss_d.backward()
            self.optimizer["D"].step()

            self.ema.update(self.model)

            return loss_d + loss_g

        else:
            scaled_loss = loss / self.gradient_accumulation_steps
            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            del scaled_loss

            return loss

    def train_epoch(self, dataloader):
        """
        Returns the epoch mean loss
        """
        self.model.train()
        total_loss = torch.tensor([0.0], device=self.device)
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            loss = self.train_step(batch)
            total_loss += loss.detach()

            if self.config.get("model_type") == "ddgan":
                continue

            is_update_step = (step + 1) % self.gradient_accumulation_steps == 0 or (
                step + 1
            ) == len(dataloader)

            if is_update_step:
                if self.grad_offloader is None:
                    if self.scaler:
                        if self.grad_clip > 0.0:
                            self.scaler.unscale_(self.optimizer)
                            norm_unet = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.optimizer.step()
                else:
                    self.grad_offloader.finalize_and_step(
                        self.optimizer, scaler=self.scaler, max_norm=self.grad_clip
                    )

                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

                self.ema.update(self.model)

        return total_loss.item() / len(dataloader)

    def train(
        self,
        epochs,
        dataloader,
        log_interval=1,
        sample_interval=0,
        save_interval=0,
        timestamp=None,
        num_steps=50,
    ):
        """
        Trains for the given epochs and prints the loss in the interval
        """

        if sample_interval == 0:
            sample_interval = epochs + 1

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.info(
            f"Starting training: {self.config['schedule_type']} | Pred: {self.prediction_target}"
        )

        output_dir = self.config.get("output_dir", f"results/checkpoints/{timestamp}")

        for epoch in range(self.start_epoch, epochs):
            if (
                self.use_ema_config
                and epoch >= self.ema_start_epoch
                and self.ema.ema_model is None
            ):
                logging.info(f"Initializing EMA model at epoch {epoch}")
                self.ema.initialize(self.model)

            loss = self.train_epoch(dataloader)

            if (epoch + 1) % log_interval == 0:
                if isinstance(self.optimizer, dict):
                    lr_val = self.optimizer["G"].param_groups[0]["lr"]
                else:
                    lr_val = self.optimizer.param_groups[0]["lr"]

                logging.info(
                    f"[Epoch {epoch + 1}/{epochs}] Loss: {loss:.6f} LR: {lr_val:.3e}"
                )

            if (epoch + 1) % sample_interval == 0:
                self.run_sampling(timestamp, epoch, num_steps)

            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    ema=self.ema,
                )

        return self.model

    def run_sampling(self, timestamp, epoch, num_steps=50):
        """
        Helper to run sampling and save visualizations.
        """
        return_traj = not self.is_latents

        sample_result = self.sample(
            self.prediction_target, 16, num_steps=num_steps, return_traj=return_traj
        )
        torch.cuda.empty_cache()

        if return_traj:
            final_samples, traj = sample_result
        else:
            final_samples, traj = sample_result, None

        save_path_grid = f"results/images/{timestamp}/ep{epoch}_grid.png"
        visualize_image_grid(final_samples, save_path_grid, nrow=4)

        if traj is not None and self.config.get("loss_target") != "consistency":
            save_path_traj = f"results/images/{timestamp}/ep{epoch}_traj.png"
            visualize_image_trajectory(traj, save_path_traj, num_rows=4)

    def sample(
        self,
        pred_target,
        num_sampling,
        data_shape=None,
        num_steps=50,
        return_traj=True,
        vae_batch_size=32,
        prompts=None,
        filter_type="none",
        keep_ratio=1.0,
        var_timesteps=15,
    ):
        self.model.eval()
        torch.cuda.empty_cache()

        if data_shape is None:
            if hasattr(self.dataset, "P"):
                D = self.dataset[0].shape[-1]
                # Reflow dataset
                if isinstance(self.dataset[0], (tuple, list)):
                    D = self.dataset[0][0].shape[-1]
            else:
                # Image self.dataset
                if isinstance(self.dataset[0], (tuple, list)):
                    D = list(self.dataset[0][0].shape)
                else:
                    D = list(self.dataset[0].shape)  # (C, H, W)

        else:
            D = data_shape

        projection_matrix = getattr(self.dataset, "P", None)

        embeddings = None
        attention_mask = None
        cfg_scale = self.config.get("sampling", {}).get("cfg_scale", 1.0)

        if self.conditional:
            if prompts is not None:
                assert len(prompts) == num_sampling
                neg_prompts = [""] * num_sampling
                full_prompts = neg_prompts + prompts if cfg_scale > 1.0 else prompts
            else:
                prompts_list = [""] * num_sampling
                neg_prompts = [""] * num_sampling
                if self.sample_configs and num_sampling <= len(self.sample_configs):
                    prompts_list = [
                        c.get("prompt", "") for c in self.sample_configs[:num_sampling]
                    ]
                    neg_prompts = [
                        c.get("negative_prompt", "")
                        for c in self.sample_configs[:num_sampling]
                    ]
                full_prompts = (
                    neg_prompts + prompts_list if cfg_scale > 1.0 else prompts_list
                )

            with torch.no_grad():
                embeddings, attention_mask = self.model["text_enc"](full_prompts)

        force_traj = filter_type != "none"

        with self.ema.average_parameters(self.model):
            with torch.autocast(
                device_type=self.device,
                dtype=self.autocast_dtype,
                enabled=self.autocast_enabled,
            ):
                samples = generate_samples(
                    model=self.model,
                    schedule=self.schedule,
                    batch_size=num_sampling,
                    data_shape=D,
                    diffusion_type=self.config.get("schedule_type", "linear"),
                    prediction_target=pred_target,
                    num_steps=num_steps,
                    cfg_scale=cfg_scale,
                    embeddings=embeddings,
                    attention_mask=attention_mask,
                    is_conditional=self.conditional,
                    projection_matrix=projection_matrix,
                    return_traj=return_traj or force_traj,
                    vae=self.vae,
                    vae_scale=self.config.get("vae_scale", 1.0),
                    vae_shift=self.config.get("vae_shift", 0.0),
                    vae_batch_size=vae_batch_size,
                    sampler_type=self.config.get("sampler_type", "ddim"),
                    shift=self.config.get("sample_shift", 1.0),
                    clip_prediction=self.config.get("clip_prediction", False),
                    cm_steps=self.config.get("cm_steps", 4),
                    ddgan_steps=self.config.get("ddgan_steps", 4),
                )

        if return_traj or force_traj:
            # sample functions return (final, traj) if return_traj=True
            final_samples, traj = samples
        else:
            final_samples = samples
            traj = None

        variance_scores = None

        if filter_type != "none":
            num_keep = int(num_sampling * keep_ratio)

            if filter_type == "variance":
                # trajs have shape (Steps, Batch, C, H, W)
                recent_traj = traj[-var_timesteps:, :]
                var_per_dim = np.var(recent_traj, axis=1)
                axes_to_mean = tuple(range(1, var_per_dim.ndim))
                variance_scores = np.mean(var_per_dim, axis=axes_to_mean)

                if keep_ratio < 1.0:
                    sorted_indices = np.argsort(variance_scores)
                    filtered_indices = sorted_indices[:num_keep]

                    final_samples = final_samples[filtered_indices]
                    if return_traj or force_traj:
                        traj = traj[filtered_indices]
                    variance_scores = variance_scores[filtered_indices]

            elif filter_type == "random":
                variance_scores = np.zeros(num_sampling)
                if keep_ratio < 1.0:
                    filtered_indices = random.sample(range(num_sampling), num_keep)
                    final_samples = final_samples[filtered_indices]
                    if return_traj or force_traj:
                        traj = traj[filtered_indices]
                    variance_scores = variance_scores[filtered_indices]

        if filter_type != "none":
            if return_traj:
                return final_samples, traj, variance_scores
            else:
                return final_samples, variance_scores
        else:
            if return_traj:
                return final_samples, traj
            else:
                return final_samples
