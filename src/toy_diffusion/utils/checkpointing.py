import os
import json
import logging
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file


def save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    ema=None,
    config: dict = None,
    vocab: dict = None,
    skip_text_enc: bool = False,
):
    """
    Saves model checkpoint weights, optimizer, scheduler, EMA,
    config.json (HuggingFace architecture style), and vocab.json.
    """

    save_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if skip_text_enc:
        clean_state_dict = {
            k: v for k, v in clean_state_dict.items()
            if not k.startswith("text_enc.")
        }

    # Cast only floating-point tensors to bfloat16 to halve disk size
    clean_state_dict = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in clean_state_dict.items()
    }

    model_path = os.path.join(save_dir, "model.safetensors")
    save_file(clean_state_dict, model_path)

    if ema is not None and ema.use_ema and ema.ema_model is not None:
        ema_state_dict = ema.ema_model.state_dict()
        clean_ema_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in ema_state_dict.items()
        }

        if skip_text_enc:
            clean_ema_state_dict = {
                k: v for k, v in clean_ema_state_dict.items()
                if not k.startswith("text_enc.")
            }

        # Cast EMA floating point tensors
        clean_ema_state_dict = {
            k: v.to(torch.bfloat16) if v.is_floating_point() else v
            for k, v in clean_ema_state_dict.items()
        }
        ema_path = os.path.join(save_dir, "ema_model.safetensors")
        save_file(clean_ema_state_dict, ema_path)

    torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))

    # 4. Save Config JSON (HuggingFace Style)
    if config is not None:
        hf_config = {
            "_class_name": config.get("model_type", "unet"),
            "model_type": config.get("model_type", "unet"),
            "in_channels": config.get("in_channels", 3),
            "out_channels": config.get("out_channels", 3),
            "hidden_dim": config.get("hidden_dim", 128),
            "num_layers": config.get("num_layers", 3),
            "ch_mult": config.get("ch_mult", 2),
            "cross_attention_dim": config.get("cross_attention_dim", 256),
            "is_conditional": config.get("is_conditional", False),
            "is_latents": config.get("is_latents", False),
            "vae_scale": config.get("vae_scale", 1.0),
            "vae_shift": config.get("vae_shift", 0.0),
            "schedule_type": config.get("schedule_type", "linear"),
            "prediction_target": config.get("prediction_target", "v"),
            "loss_target": config.get("loss_target", "v"),
            "tiers_len": config.get("tiers_len", [24, 52]),
            "max_seq_len": config.get("max_seq_len", 16),
        }

        # Include other serializable configuration items
        for k, v in config.items():
            if k not in hf_config and isinstance(
                v, (int, float, str, bool, list, dict)
            ):
                hf_config[k] = v

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)

    # 5. Save Vocab JSON
    if vocab is None and config is not None:
        vocab = config.get("vocab", None)

    if vocab is not None:
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
    logging.info(f"Checkpoint saved successfully at {save_dir}")


def load_checkpoint_config(checkpoint_dir: str) -> dict:
    """Loads config.json from checkpoint directory if present."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_checkpoint_vocab(checkpoint_dir: str) -> dict:
    """Loads vocab.json from checkpoint directory if present."""
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_from_checkpoint(
    checkpoint_dir: str,
    model: nn.Module = None,
    optimizer=None,
    scheduler=None,
    ema=None,
    skip_text_enc: bool = False,
) -> tuple[int, dict, dict]:
    """
    Loads states from a checkpoint directory.
    Returns tuple of (start_epoch, config, vocab).
    """
    logging.info(f"Loading checkpoint from {checkpoint_dir}")

    strict = True if not skip_text_enc else False
    if model is not None:
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            sanitized_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(sanitized_dict, strict=strict)

    if ema is not None and getattr(ema, "use_ema", False):
        ema_path = os.path.join(checkpoint_dir, "ema_model.safetensors")
        if os.path.exists(ema_path):
            if ema.ema_model is None and model is not None:
                ema.initialize(model)
            if ema.ema_model is not None:
                ema.ema_model.load_state_dict(load_file(ema_path), strict=strict)

    if optimizer is not None:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu")
            if isinstance(optimizer, dict) and isinstance(opt_state, dict):
                for k, opt in optimizer.items():
                    if k in opt_state:
                        opt.load_state_dict(opt_state[k])
            else:
                optimizer.load_state_dict(opt_state)

    if scheduler is not None:
        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            try:
                scheduler.load_state_dict(
                    torch.load(sched_path, map_location="cpu")
                )
            except Exception as e:
                logging.error(
                    f"Could not load scheduler state dict: {e}. "
                    "Proceeding with initialized scheduler."
                )

    start_epoch = 0
    base_name = os.path.basename(os.path.normpath(checkpoint_dir))
    if base_name.startswith("epoch_"):
        try:
            start_epoch = int(base_name.split("_")[1])
        except ValueError:
            pass

    ckpt_config = load_checkpoint_config(checkpoint_dir)
    ckpt_vocab = load_checkpoint_vocab(checkpoint_dir)

    logging.info(f"Resumed training from epoch {start_epoch}")
    return start_epoch, ckpt_config, ckpt_vocab
