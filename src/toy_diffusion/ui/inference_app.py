import os
import json
import time
import math
import random
import torch
import numpy as np
import gradio as gr
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from toy_diffusion.paths.sampling import generate_samples

def make_image_grid(images, rows=None, cols=None):
    """Combines a list of PIL Images into a single grid Image."""
    if not images:
        return None
    n = len(images)
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    if rows is None:
        rows = int(math.ceil(n / cols))
        
    w, h = images[0].size
    grid_w = cols * w
    grid_h = rows * h
    grid_img = Image.new("RGB", (grid_w, grid_h))
    
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid_img.paste(img, (x, y))
    return grid_img

def generate_images_custom(
    trainer,
    prompt,
    neg_prompt,
    num_samples,
    steps,
    cfg_scale,
    batch_size,
    seed,
    output_dir=None,
):
    """Custom batch generator supporting prompt saving and metadata."""
    if seed is not None and seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    device = trainer.device
    is_conditional = trainer.conditional
    
    if hasattr(trainer.dataset, "P") and trainer.dataset.P is not None:
        D = trainer.dataset[0].shape[-1]
        if isinstance(trainer.dataset[0], (tuple, list)):
            D = trainer.dataset[0][0].shape[-1]
        data_shape = (D,)
    else:
        if isinstance(trainer.dataset[0], (tuple, list)):
            data_shape = list(trainer.dataset[0][0].shape)
        else:
            data_shape = list(trainer.dataset[0].shape)

    embeddings = None
    attention_mask = None
    if is_conditional:
        if "text_enc" in trainer.model:
            text_enc = getattr(
                trainer.model["text_enc"],
                "_orig_mod",
                trainer.model["text_enc"],
            )
            text_enc.shuffle = False
            text_enc.cfg_dropout_prob = 0.0
            text_enc.tag_dropout_prob = 0.0
            
        positive_prompts = [prompt] * num_samples
        negative_prompts = [neg_prompt] * num_samples
        full_prompts = (
            negative_prompts + positive_prompts
            if cfg_scale > 1.0
            else positive_prompts
        )
        
        with torch.no_grad():
            embeddings, attention_mask = trainer.model["text_enc"](
                full_prompts
            )

    trainer.model.eval()
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        curr_batch_size = min(batch_size, num_samples - i)
        curr_embeddings = None
        curr_attention_mask = None
        
        if is_conditional and embeddings is not None:
            if cfg_scale > 1.0:
                neg_slice = embeddings[:num_samples][i : i + curr_batch_size]
                pos_slice = embeddings[num_samples:][i : i + curr_batch_size]
                curr_embeddings = torch.cat([neg_slice, pos_slice], dim=0)
                
                neg_mask_slice = (
                    attention_mask[:num_samples]
                    [i : i + curr_batch_size]
                )
                pos_mask_slice = (
                    attention_mask[num_samples:]
                    [i : i + curr_batch_size]
                )
                curr_attention_mask = torch.cat(
                    [neg_mask_slice, pos_mask_slice], dim=0
                )
            else:
                curr_embeddings = embeddings[i : i + curr_batch_size]
                curr_attention_mask = (
                    attention_mask[i : i + curr_batch_size]
                )
                
        with torch.no_grad():
            with torch.autocast(
                device_type=device,
                dtype=trainer.autocast_dtype,
                enabled=trainer.autocast_enabled,
            ):
                samples = generate_samples(
                    model=trainer.model,
                    schedule=trainer.schedule,
                    batch_size=curr_batch_size,
                    data_shape=data_shape,
                    x=None,
                    diffusion_type=trainer.config.get(
                        "schedule_type", "linear"
                    ),
                    prediction_target=trainer.prediction_target,
                    num_steps=steps,
                    cfg_scale=cfg_scale,
                    embeddings=curr_embeddings,
                    attention_mask=curr_attention_mask,
                    is_conditional=is_conditional,
                    projection_matrix=getattr(trainer.dataset, "P", None),
                    return_traj=False,
                    vae=trainer.vae,
                    device=device,
                    vae_scale=trainer.config.get("vae_scale", 1.0),
                    vae_shift=trainer.config.get("vae_shift", 0.0),
                    vae_batch_size=32,
                    sampler_type=trainer.config.get("sampler_type", "ddim"),
                    shift=trainer.config.get("sample_shift", 1.0),
                    clip_prediction=trainer.config.get(
                        "clip_prediction", False
                    ),
                )
                
        samples = np.clip((samples + 1.0) / 2.0, 0.0, 1.0)
        if samples.ndim == 4 and samples.shape[1] in [1, 3, 4]:
            samples = np.transpose(samples, (0, 2, 3, 1))
            
        for j in range(samples.shape[0]):
            img_np = (samples[j] * 255.0).astype(np.uint8)
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            all_images.append(Image.fromarray(img_np))
            
    grid_img = make_image_grid(all_images)
    
    # Save individual and grid images with metadata
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        
        meta_payload = {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "checkpoint": (
                trainer.config.get("resume_from_checkpoint") or "base"
            ),
            "schedule_type": trainer.config.get("schedule_type", "linear"),
            "prediction_target": trainer.prediction_target,
        }
        meta_str = json.dumps(meta_payload)
        png_info = PngInfo()
        png_info.add_text("parameters", meta_str)
        
        for idx, img in enumerate(all_images):
            img_path = os.path.join(
                output_dir, f"sample_{timestamp}_{idx}.png"
            )
            img.save(img_path, pnginfo=png_info)
            
        if grid_img is not None:
            grid_path = os.path.join(
                output_dir, f"grid_{timestamp}.png"
            )
            grid_img.save(grid_path, pnginfo=png_info)
            
    return [grid_img] + all_images if grid_img is not None else all_images

def read_metadata(image):
    """Helper to parse custom json parameter string from PNG metadata."""
    if image is None:
        return "Upload an image to see metadata."
    try:
        parameters = image.info.get("parameters", "")
        if not parameters:
            return "No metadata found in this image."
        data = json.loads(parameters)
        return json.dumps(data, indent=4)
    except Exception:
        raw_data = image.info.get("parameters", "None")
        return f"Raw parameters:\n{raw_data}"

def send_to_generation(image):
    """Parses data dictionary to restore UI values."""
    if image is None:
        return [gr.skip()] * 5
    try:
        parameters = image.info.get("parameters", "")
        if not parameters:
            return [gr.skip()] * 5
        data = json.loads(parameters)
        
        p = data.get("prompt", gr.skip())
        np_ = data.get("neg_prompt", gr.skip())
        s = data.get("steps", gr.skip())
        c = data.get("cfg_scale", gr.skip())
        sd = data.get("seed", gr.skip())
        
        return [p, np_, s, c, sd]
    except Exception:
        return [gr.skip()] * 5

def create_inference_ui(
    trainer,
    default_prompt,
    default_neg_prompt,
    default_steps,
    default_cfg,
    default_batch_size,
    output_dir,
):
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

    with gr.Blocks(
        title="Diffusion Inference", theme=theme, js=js_func
    ) as demo:
        gr.Markdown("# 🌌 Text-to-Image Generation (Flow Matching vs DDPM)")

        with gr.Tabs():
            with gr.Tab("Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt = gr.Textbox(
                            value=default_prompt,
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=3,
                        )
                        neg_prompt = gr.Textbox(
                            value=default_neg_prompt,
                            label="Negative Prompt",
                            placeholder="Enter your negative prompt here...",
                            lines=2,
                        )
                        
                        with gr.Row():
                            steps = gr.Slider(
                                minimum=1,
                                maximum=150,
                                value=default_steps,
                                step=1,
                                label="Steps",
                            )
                            cfg = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                value=default_cfg,
                                step=0.5,
                                label="CFG Scale",
                            )
                        
                        with gr.Row():
                            num_samples = gr.Slider(
                                minimum=1,
                                maximum=32,
                                value=4,
                                step=1,
                                label="Number of Samples",
                            )
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=16,
                                value=default_batch_size,
                                step=1,
                                label="Batch Size",
                            )
                        
                        seed = gr.Number(
                            value=-1,
                            label="Seed (-1 for random)",
                            precision=0,
                        )
                        
                        output_dir = gr.Textbox(
                            value=output_dir,
                            label="Output Directory",
                            placeholder="results/inference",
                        )
                        
                        generate_btn = gr.Button(
                            "Generate Images", variant="primary"
                        )

                    with gr.Column(scale=1):
                        gallery = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            columns=2,
                            rows=2,
                            object_fit="contain",
                            height="auto",
                        )

            with gr.Tab("PNG Info"):
                with gr.Row():
                    with gr.Column(scale=1):
                        info_image = gr.Image(
                            type="pil",
                            label="Upload Image",
                        )
                    with gr.Column(scale=1):
                        metadata_text = gr.Textbox(
                            label="Metadata Details",
                            interactive=False,
                            lines=12,
                        )
                        send_btn = gr.Button("Send to Generation")

        def generate_fn(
            prompt_val,
            neg_prompt_val,
            num_samples_val,
            steps_val,
            cfg_scale_val,
            batch_size_val,
            seed_val,
            out_dir_val,
        ):
            return generate_images_custom(
                trainer=trainer,
                prompt=prompt_val,
                neg_prompt=neg_prompt_val,
                num_samples=int(num_samples_val),
                steps=int(steps_val),
                cfg_scale=float(cfg_scale_val),
                batch_size=int(batch_size_val),
                seed=int(seed_val),
                output_dir=out_dir_val,
            )

        generate_btn.click(
            fn=generate_fn,
            inputs=[
                prompt,
                neg_prompt,
                num_samples,
                steps,
                cfg,
                batch_size,
                seed,
                output_dir,
            ],
            outputs=[gallery],
        )

        # Metadata dynamic handlers
        info_image.change(
            fn=read_metadata,
            inputs=[info_image],
            outputs=[metadata_text],
        )

        send_btn.click(
            fn=send_to_generation,
            inputs=[info_image],
            outputs=[prompt, neg_prompt, steps, cfg, seed],
        )

    return demo