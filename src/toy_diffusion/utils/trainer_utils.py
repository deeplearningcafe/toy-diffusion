import torch
import torch.nn as nn
import logging
import omegaconf
import copy
import contextlib
import os
from safetensors.torch import save_file, load_file

try:
    from diffusers import AutoencoderKL
except ImportError:
    print("Diffusers is not installed")

from toy_diffusion.models.edm_model import EDMPreconditioner
from toy_diffusion.models.cm import ConsistencyPreconditioner
from toy_diffusion.models.mlp import (
    ResModel,
    FlowMLP,
    SimpleNet,
    DDGANGenerator,
    DDGANDiscriminator,
)
from toy_diffusion.models.unet import Unet
from toy_diffusion.models.efficient_unet import EfficientUnet
from toy_diffusion.models.dual_stream import DualStreamDiT
from toy_diffusion.models.aux_models import (
    EMAModel,
    SimpleTextEncoder,
    init_weights,
    HFTextEncoder,
)
from toy_diffusion.models.sprint import (
    SprintLuminaNextDit,
    SprintDualStreamDiT,
)
from toy_diffusion.losses import (
    GeneralDiffusionLoss,
    EDMLoss,
    ConsistencyTrainingLoss,
    DDGANLoss,
)
from toy_diffusion.paths.scheduler import LinearSchedule, DDPMSchedule, VESchedule


def get_model(config, device):
    text_enc = None
    cross_attention_dim = config.get("cross_attention_dim", None)

    if config.get("is_conditional", False):
        hf_model_id = config.get("hf_text_encoder", None)
        if hf_model_id:
            text_enc = HFTextEncoder(
                model_id=hf_model_id,
                max_seq_len=config.get("max_seq_len", 256),
            ).to(device)
            cross_attention_dim = text_enc.embed_dim
        else:
            vocab = config.get("vocab", {"<pad>": 0, "<unk>": 1})
            max_seq_len = config.get("max_seq_len", 16)
            if cross_attention_dim is None:
                cross_attention_dim = 256
            text_enc = SimpleTextEncoder(
                vocab=vocab,
                max_seq_len=max_seq_len,
                embed_dim=cross_attention_dim,
                tiers_len=config["tiers_len"],
                use_pos=config.get("use_pos", False),
            ).to(device)

    if config["model_type"] == "resnet":
        model = ResModel(
            data_dim=config["projection_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"] if config["dataset_type"] != "nerv" else 10,
            activation=torch.nn.SiLU,
        ).to(device)
    elif config["model_type"] == "simple_net":
        dim_hids = [config["hidden_dim"] for _ in range(config["num_layers"])]
        model = SimpleNet(
            dim_in=config["projection_dim"],
            dim_out=config["projection_dim"],
            dim_hids=dim_hids,
            num_timesteps=1000,
        ).to(device)
        model.apply(init_weights)
    elif config["model_type"] == "unet":
        # pixel space 3 channels, latents 4,16 or 32
        block_out_channels = [64, 128, 256, 256]
        in_channels = config.get("in_channels", 3)
        unet = Unet(
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=[
                item * config["ch_mult"] for item in block_out_channels
            ],
            norm_num_groups=32,
            cross_attention_dim=cross_attention_dim,
            device=device,
            use_checkpointing=config.get("use_gradient_checkpointing", False),
            dropout=config.get("dropout", 0.1),
        ).to(device)

        model = (
            nn.ModuleDict({"unet": unet, "text_enc": text_enc})
            if text_enc is not None
            else unet
        )
    elif config["model_type"] == "efficient_unet":
        in_channels = config.get("in_channels", 3)
        unet = EfficientUnet(
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            device=device,
            use_checkpointing=config.get("use_gradient_checkpointing", False),
            dropout=config.get("dropout", 0.1),
        ).to(device)

        model = (
            nn.ModuleDict({"unet": unet, "text_enc": text_enc})
            if text_enc is not None
            else unet
        )
    elif config["model_type"] == "dual_stream":
        in_channels = config.get("in_channels", 3)
        unet = DualStreamDiT(
            in_channels=in_channels,
            out_channels=in_channels,
            hidden_size=config["hidden_dim"],
            num_heads=config.get("num_heads", 12),
            text_embed_dim=cross_attention_dim,
            depth=config["depth"],
            use_checkpointing=config.get("use_gradient_checkpointing", False),
        ).to(device)

        model = (
            nn.ModuleDict({"unet": unet, "text_enc": text_enc})
            if text_enc is not None
            else unet
        )
    elif config["model_type"] in ["sprint_single", "sprint_dual"]:
        in_channels = config.get("in_channels", 3)
        if config["model_type"] == "sprint_single":
            unet = SprintLuminaNextDit(
                in_channels=in_channels,
                hidden_size=config["hidden_dim"],
                depth=config["depth"],
                num_attention_heads=config.get("num_heads", 16),
                cross_attention_dim=cross_attention_dim,
                encoder_depth=config.get("encoder_depth", 2),
                decoder_depth=config.get("decoder_depth", 2),
                drop_ratio=config.get("drop_ratio", 0.75),
                residual_type=config.get("residual_type", "concat_linear"),
                cfg_mask_prob=config.get("cfg_mask_prob", 0.1),
            ).to(device)
        else:
            unet = SprintDualStreamDiT(
                in_channels=in_channels,
                out_channels=in_channels,
                hidden_size=config["hidden_dim"],
                depth=config["depth"],
                num_heads=config.get("num_heads", 12),
                text_embed_dim=cross_attention_dim,
                use_checkpointing=config.get("use_gradient_checkpointing", False),
                encoder_depth=config.get("encoder_depth", 2),
                decoder_depth=config.get("decoder_depth", 2),
                drop_ratio=config.get("drop_ratio", 0.75),
                drop_target=config.get("drop_target", "image"),
                residual_type=config.get("residual_type", "concat_linear"),
                cfg_mask_prob=config.get("cfg_mask_prob", 0.1),
            ).to(device)

        model = (
            nn.ModuleDict({"unet": unet, "text_enc": text_enc})
            if text_enc is not None
            else unet
        )

    elif config.get("model_type") == "ddgan":
        model = nn.ModuleDict(
            {
                "G": DDGANGenerator(
                    data_dim=config.get("projection_dim", 2),
                    latent_dim=config.get("latent_dim", 4),
                ),
                "D": DDGANDiscriminator(data_dim=config.get("projection_dim", 2)),
            }
        ).to(device)
    else:
        model = FlowMLP(
            data_dim=config["projection_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"] if config["dataset_type"] != "nerv" else 10,
            activation=torch.nn.SiLU,
        ).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Trainable parameters in model: {count_parameters(model) / 1e6} M")
    return model


def get_schedule_loss(config, model, prediction_target, device):
    # DD-GAN uses a specific scheduler and loss
    if config.get("model_type") == "ddgan":
        print("Using dd-gan scheduler and loss")
        schedule = DDPMSchedule(device=device)
        loss_fn = DDGANLoss(
            schedule=schedule, num_timesteps=config.get("ddgan_steps", 4)
        )
        return schedule, loss_fn, model

    if config["loss_target"] == "consistency":
        print("Using improved consistency scheduler (iCT)")
        schedule = VESchedule(device=device)
        model = ConsistencyPreconditioner(model)

        total_steps = config.get("total_training_steps", 10000)
        loss_fn = ConsistencyTrainingLoss(total_training_steps=total_steps)
        return schedule, loss_fn, model

    elif config["schedule_type"] == "linear":
        print("Using linear scheduler")
        schedule = LinearSchedule(device=device)

    elif config["loss_target"] == "karras_edm":
        print("Using karras scheduler")
        schedule = VESchedule(device=device)
        # loss target is always the denoised image
        loss_fn = EDMLoss()
        model = EDMPreconditioner(model, prediction_target=prediction_target)
    else:
        print("Using ddpm scheduler")
        schedule = DDPMSchedule(device=device, beta_start=0.00001, beta_end=0.02)

    if config["loss_target"] not in ["karras_edm", "consistency"]:
        # Allow config to explicitly set 0.0 to disable it.
        default_perturb = 0.1 if config["schedule_type"] == "ddpm" else 0.0
        input_perturbation = config.get("input_perturbation", default_perturb)

        default_weight_fn = (
            "min-snr-gamma" if config["schedule_type"] == "ddpm" else None
        )
        weight_fn_name = config.get("weight_fn_name", default_weight_fn)
        timestep_sampling = config.get("timestep_fn", "uniform")

        loss_fn = GeneralDiffusionLoss(
            schedule=schedule,
            prediction_target=prediction_target,
            loss_target=config["loss_target"],
            timestep_sampling=timestep_sampling,
            weight_fn_name=weight_fn_name,
            input_perturbation=input_perturbation,
            use_ot=config.get("use_ot", False),
            train_shift=config.get("train_shift", 1.0),
            is_conditional=config.get("is_conditional", False),
        )
    return schedule, loss_fn, model


def create_optimizer_param_groups(
    model,
    lr,
    weight_decay,
):
    param_groups = []
    no_decay_keywords = ["bias", "norm"]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]

        if any(k in name for k in no_decay_keywords):
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups.append(
        {
            "params": decay,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": "decay",
        }
    )
    param_groups.append(
        {
            "params": no_decay,
            "lr": lr,
            "weight_decay": 0.0,
            "name": "no_decay",
        }
    )
    return param_groups


def create_optim_scheduler(model, len_train_loader: int, conf: omegaconf.DictConfig):
    if conf.get("model_type") == "ddgan":
        optimizer_g = torch.optim.Adam(
            model["G"].parameters(), lr=conf.get("lr", 1e-4), betas=(0.5, 0.9)
        )
        optimizer_d = torch.optim.Adam(
            model["D"].parameters(), lr=conf.get("lr", 1e-4), betas=(0.5, 0.9)
        )
        return {"G": optimizer_g, "D": optimizer_d}, None

    param_groups = create_optimizer_param_groups(
        model, conf["lr"], conf.get("wd", 0.01)
    )
    if conf.get("use_bitsandbytes", False):
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=conf["lr"],
            betas=(0.9, 0.98),
        )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=conf["lr"],
            betas=(0.9, 0.98),
        )

    scheduler = None
    total_steps = conf["epochs"] * len_train_loader
    warmup_steps = int(conf.get("warmup", 0.02) * total_steps)
    warmup_steps = min(warmup_steps, total_steps - 1) if total_steps > 0 else 0

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )

    if conf.get("use_cos_scheduler", False):
        print("Using cosine lr scheduler")
        cosine_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_steps, eta_min=conf["lr"] * 0.1
        )
    else:
        print("Using constant lr scheduler")
        constant_steps = max(1, total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=constant_steps
        )

    if warmup_steps > 0:
        return optimizer, torch.optim.lr_scheduler.SequentialLR(
            optimizer, [scheduler_warmup, scheduler], milestones=[warmup_steps]
        )

    return optimizer, scheduler


def save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    ema: EMAModel = None,
):
    """Saves model, optimizer, scheduler, and EMA states safely."""
    save_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model_path = os.path.join(save_dir, "model.safetensors")
    save_file(clean_state_dict, model_path)

    if ema is not None and ema.use_ema and ema.ema_model is not None:
        ema_state_dict = ema.ema_model.state_dict()
        clean_ema_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in ema_state_dict.items()
        }
        ema_path = os.path.join(save_dir, "ema_model.safetensors")
        save_file(clean_ema_state_dict, ema_path)

    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    logging.info(f"Checkpoint saved successfully at {save_dir}")


def load_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    ema: EMAModel = None,
) -> int:
    """Loads states from a checkpoint directory and returns the start epoch."""
    logging.info(f"Loading checkpoint from {checkpoint_dir}")

    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)
        sanitized_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
        }
        model.load_state_dict(sanitized_state_dict)

    ema_path = os.path.join(checkpoint_dir, "ema_model.safetensors")
    if ema is not None and ema.use_ema and os.path.exists(ema_path):
        if ema.ema_model is None:
            ema.initialize(model)
        ema.ema_model.load_state_dict(load_file(ema_path))

    opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))

    sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
    if scheduler is not None and os.path.exists(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

    # Extract epoch from directory name (e.g., "epoch_10")
    start_epoch = 0
    base_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if base_name.startswith("epoch_"):
        try:
            start_epoch = int(base_name.split("_")[1])
        except ValueError:
            pass

    logging.info(f"Resuming training from epoch {start_epoch}")
    return start_epoch


def gpu_setup(device: str = "cuda"):
    autocast_dtype = torch.float32
    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        autocast_dtype = torch.bfloat16
        if capability[0] < 8:
            print(
                f"Warning: bfloat16 specified but GPU capability "
                f"({capability[0]}.{capability[1]}) may not fully support it. "
                f"Consider float16 or float32."
            )
            autocast_dtype = torch.float32

        if capability[0] >= 7 and capability[0] < 8:
            autocast_dtype = torch.float16
            torch.set_float32_matmul_precision("high")
            print("Using high precision for float32 matmul (tensor cores).")
        elif capability[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            print("Using half precision for float32 matmul (tensor cores).")
        else:
            print(
                "Tensor cores for float32 matmul not optimally supported or GPU is older."
            )
    elif device == "xpu":
        print(f"Using xpu device with torch version {torch.__version__}")
        autocast_dtype = torch.bfloat16
    return autocast_dtype


def get_vae(config, device, dtype=torch.bfloat16):
    """Loads a pretrained VAE and wraps it for latent decoding."""
    if not config.get("is_latents", False):
        return None

    vae_pretrained = config.get("vae_pretrained", "black-forest-labs/FLUX.1-dev")
    logging.info(f"Loading pretrained VAE: {vae_pretrained} for latent decoding...")

    vae = AutoencoderKL.from_pretrained(vae_pretrained, torch_dtype=dtype).to(device)

    vae.eval()
    vae.requires_grad_(False)

    return vae
