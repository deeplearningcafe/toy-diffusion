import os
import json
import logging
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

PIXEL_EXPLICIT_MODULES = {
    "x_embedder",
    "conv_in",
    "pixel_decoder",
    "proj_out",
    "conv_out",
    "norm_final",
    "norm_out",
}


def _normalize_param_key(k: str) -> str:
    """Removes torch.compile wrappers and module prefixes for matching."""
    k = k.replace("_orig_mod.", "")
    if k.startswith("unet."):
        k = k[5:]
    return k


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
    train_output_only: bool = False,
):
    """
    Saves model checkpoint weights, optimizer, scheduler, EMA,
    config.json, and vocab.json. If train_output_only is True, saves
    only explicitly defined pixel adaptation layers.
    """

    save_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if skip_text_enc:
        clean_state_dict = {
            k: v for k, v in clean_state_dict.items() if not k.startswith("text_enc.")
        }

    is_output_only = train_output_only or (
        config is not None and config.get("train_output_only", False)
    )
    if is_output_only:
        clean_state_dict = {
            k: v
            for k, v in clean_state_dict.items()
            if set(k.split(".")) & PIXEL_EXPLICIT_MODULES
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
                k: v
                for k, v in clean_ema_state_dict.items()
                if not k.startswith("text_enc.")
            }

        if is_output_only:
            clean_ema_state_dict = {
                k: v
                for k, v in clean_ema_state_dict.items()
                if set(k.split(".")) & PIXEL_EXPLICIT_MODULES
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
            "train_output_only": train_output_only,
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
    pixel_dir: str = None,
) -> tuple[int, dict, dict]:
    """
    Loads states from a checkpoint directory.
    Returns tuple of (start_epoch, config, vocab).
    """
    logging.info(f"Loading checkpoint from {checkpoint_dir}")

    strict = True if (not skip_text_enc and pixel_dir is None) else False
    if model is not None:
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(model_path):
            state_dict = load_file(model_path)
            target_state = model.state_dict()
            target_map = {_normalize_param_key(k): k for k in target_state.keys()}
            sanitized_dict = {}
            for k, v in state_dict.items():
                norm_k = _normalize_param_key(k)
                if norm_k in target_map:
                    real_k = target_map[norm_k]
                    if target_state[real_k].shape == v.shape:
                        sanitized_dict[real_k] = v
            model.load_state_dict(sanitized_dict, strict=strict)

        if pixel_dir is not None:
            load_pixel_weights(model, pixel_dir, ema=ema)

    if ema is not None and getattr(ema, "use_ema", False):
        ema_path = os.path.join(checkpoint_dir, "ema_model.safetensors")
        if os.path.exists(ema_path):
            if ema.ema_model is None and model is not None:
                ema.initialize(model)
            if ema.ema_model is not None:
                ema_dict = load_file(ema_path)
                ema_target = ema.ema_model.state_dict()
                ema_map = {_normalize_param_key(k): k for k in ema_target.keys()}
                clean_ema = {}
                for k, v in ema_dict.items():
                    norm_k = _normalize_param_key(k)
                    if norm_k in ema_map:
                        real_k = ema_map[norm_k]
                        if ema_target[real_k].shape == v.shape:
                            clean_ema[real_k] = v
                ema.ema_model.load_state_dict(clean_ema, strict=strict)

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
                scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
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


def load_pixel_weights(
    model: nn.Module,
    checkpoint_dir: str,
    ema=None,
) -> nn.Module:
    """
    Loads only explicit pixel adaptation layers (x_embedder, pixel_decoder,
    norm_final) from a checkpoint into the model and optional EMA tracker.
    """
    logging.info(f"Loading pixel adaptation weights from: {checkpoint_dir}")
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(model_path):
        state_dict = load_file(model_path)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    target_state = model.state_dict()
    target_key_map = {_normalize_param_key(k): k for k in target_state.keys()}
    pixel_dict = {}

    for k, v in state_dict.items():
        norm_k = _normalize_param_key(k)
        parts = set(norm_k.split("."))
        if not (parts & PIXEL_EXPLICIT_MODULES):
            continue

        if norm_k in target_key_map:
            target_key = target_key_map[norm_k]
            if target_state[target_key].shape != v.shape:
                raise ValueError(
                    f"Shape mismatch for pixel layer '{target_key}': "
                    f"model {target_state[target_key].shape} vs "
                    f"ckpt {v.shape}"
                )
            pixel_dict[target_key] = v

    if not pixel_dict:
        raise KeyError(
            f"No pixel adaptation weights found in checkpoint {checkpoint_dir}."
        )

    model.load_state_dict(pixel_dict, strict=False)
    logging.info(f"Loaded {len(pixel_dict)} pixel adaptation layers successfully.")

    if ema is not None and getattr(ema, "ema_model", None) is not None:
        ema_target = ema.ema_model.state_dict()
        ema_key_map = {_normalize_param_key(k): k for k in ema_target.keys()}
        ema_path = os.path.join(checkpoint_dir, "ema_model.safetensors")
        source_dict = load_file(ema_path) if os.path.exists(ema_path) else state_dict
        if "ema_model" in source_dict:
            source_dict = source_dict["ema_model"]
        elif "model" in source_dict:
            source_dict = source_dict["model"]

        clean_ema = {}
        for k, v in source_dict.items():
            norm_k = _normalize_param_key(k)
            if set(norm_k.split(".")) & PIXEL_EXPLICIT_MODULES:
                if norm_k in ema_key_map:
                    target_k = ema_key_map[norm_k]
                    if ema_target[target_k].shape == v.shape:
                        clean_ema[target_k] = v
        if clean_ema:
            ema.ema_model.load_state_dict(clean_ema, strict=False)
        else:
            ema.ema_model.load_state_dict(pixel_dict, strict=False)

    return model
