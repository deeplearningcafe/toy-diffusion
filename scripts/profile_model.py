import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import gc

from toy_diffusion.data.image import ImageDataset, TieredBatchSampler
from toy_diffusion.utils.trainer_utils import (
    get_model,
    get_schedule_loss,
    create_optim_scheduler,
    gpu_setup,
)
from toy_diffusion.utils.profiling_utils import (
    annotate,
    annotate_model,
    flush,
    trace_handler,
)
from torch.utils.flop_counter import FlopCounterMode
from toy_diffusion.paths.sampling import generate_samples

try:
    from toy_diffusion.utils.act_grad_checkpointing import (
        patch_torch_compile,
    )
except Exception as e:
    print(f"Can't use unsloth gradient checkpoint {e}")


def run_profiling(args):
    base_conf = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_cli(args.opts)
    cfg = OmegaConf.merge(base_conf, cli_conf)

    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    config = {
        **OmegaConf.to_container(cfg.experiment),
        **OmegaConf.to_container(cfg.data),
        **OmegaConf.to_container(cfg.training),
        **OmegaConf.to_container(cfg.diffusion),
        **OmegaConf.to_container(cfg.model),
        **OmegaConf.to_container(cfg.sampling),
        "device": device,
    }

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    autocast_dtype = gpu_setup(device)

    is_conditional = config.get("is_conditional", False)
    dataset = ImageDataset(
        root_dir=config["data_path"],
        load_into_ram=args.load_into_ram,
        num_workers=config.get("num_workers", 4),
        resize_dim=config.get("resize_dim", None),
        conditional=is_conditional,
        is_latents=config.get("is_latents", False),
        shuffle_tags=config.get("shuffle_tags", False),
        cfg_dropout_prob=config.get("cfg_dropout_prob", 0.0),
        tag_dropout_prob=config.get("tag_dropout_prob", 0.0),
        use_short_prompts=config.get("use_short_prompts", False),
    )

    if config.get("is_latents"):
        config["in_channels"] = (
            dataset[0][0].shape[0] if is_conditional else dataset[0].shape[0]
        )

    if is_conditional and hasattr(dataset, "tiers"):
        batch_sampler = TieredBatchSampler(
            dataset.tiers, config["batch_size"], drop_last=True
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=config.get("num_workers", 4),
            persistent_workers=True if config.get("num_workers", 4) > 0 else False,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config.get("num_workers", 4),
            persistent_workers=True if config.get("num_workers", 4) > 0 else False,
            shuffle=True,
            pin_memory=True,
        )

    if is_conditional and hasattr(dataset, "vocab"):
        config["vocab"] = dataset.vocab
        config["max_seq_len"] = dataset.max_seq_len
        if config.get("cross_attention_dim") is None:
            config["cross_attention_dim"] = 256

    model = get_model(config, device)
    prediction_target = config.get("prediction_target", "v")
    schedule, loss_fn, model = get_schedule_loss(
        config, model, prediction_target, device
    )

    optimizer, _ = create_optim_scheduler(model, len(dataloader), config)

    # Only annotate model and loss function for trace profiling IF NOT compiled.
    if args.profile_type == "trace" and not args.compile:
        annotate_model(model)
        func = getattr(loss_fn.forward, "__func__", loss_fn.forward)
        loss_fn.forward = annotate(func, "loss_forward").__get__(loss_fn, type(loss_fn))

    if args.compile:
        print("Compiling model with torch.compile for faster training...")
        patch_torch_compile()
        if isinstance(model, torch.nn.ModuleDict) and "unet" in model:
            model["unet"] = torch.compile(model["unet"])
            # skip text_enc
        else:
            model = torch.compile(model)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == "train":
        if args.profile_type == "trace":
            profile_train_trace(
                model,
                loss_fn,
                optimizer,
                dataloader,
                device,
                output_dir,
                autocast_dtype,
                args,
            )
        elif args.profile_type == "memory_flops":
            profile_train_memory_flops(
                model, loss_fn, optimizer, dataloader, device, autocast_dtype, args
            )
    elif args.mode == "sample":
        if hasattr(dataset, "P"):
            D = dataset[0].shape[-1]
            if isinstance(dataset[0], (tuple, list)):
                D = dataset[0][0].shape[-1]
        else:
            if isinstance(dataset[0], (tuple, list)):
                D = list(dataset[0][0].shape)
            else:
                D = list(dataset[0].shape)

        if args.profile_type == "trace":
            profile_sample_trace(
                model, schedule, config, D, device, output_dir, autocast_dtype, args
            )
        elif args.profile_type == "memory_flops":
            profile_sample_memory_flops(
                model, schedule, config, D, device, autocast_dtype, args
            )


def profile_train_trace(
    model, loss_fn, optimizer, dataloader, device, output_dir, autocast_dtype, args
):
    flush()
    model.train()

    warmup_iters = args.warmup_iters
    profile_iters = args.profile_iters

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_name = "train_trace_compile" if args.compile else "train_trace"
    with_stack = not args.compile

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0, warmup=warmup_iters, active=profile_iters, repeat=1
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        on_trace_ready=lambda prof: trace_handler(prof, output_dir, trace_name),
    ) as prof:
        for i, batch in enumerate(dataloader):
            if i >= warmup_iters + profile_iters:
                break

            if isinstance(batch, (list, tuple)):
                x, prompt_tokens, prompt_mask = batch
                x = x.to(device)
                prompt = (prompt_tokens, prompt_mask)
            else:
                x = batch.to(device)
                prompt = None

            optimizer.zero_grad(set_to_none=True)

            if not args.compile:
                with torch.profiler.record_function("## forward ##"):
                    with torch.autocast(device_type=device, dtype=autocast_dtype):
                        loss = loss_fn(model, x, prompt)

                with torch.profiler.record_function("## backward ##"):
                    loss.backward()

                with torch.profiler.record_function("## optimizer ##"):
                    optimizer.step()
            else:
                # Omit record_function wrappers for compiled model to avoid graph breaks
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    loss = loss_fn(model, x, prompt)
                loss.backward()
                optimizer.step()

            prof.step()

    print(f"Trace profiling completed. Results saved to {output_dir}")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
            row_limit=20,
        )
    )


def profile_train_memory_flops(
    model, loss_fn, optimizer, dataloader, device, autocast_dtype, args
):
    flush()
    model.train()

    warmup_iters = args.warmup_iters
    profile_iters = args.profile_iters

    flops_list = []
    peak_memory_list = []
    time_list = []

    # Benchmark time and memory without FlopCounterMode
    print("Benchmarking time and memory...")
    for i, batch in enumerate(dataloader):
        if i >= warmup_iters + profile_iters:
            break

        if isinstance(batch, (list, tuple)):
            x, prompt_tokens, prompt_mask = batch
            x = x.to(device)
            prompt = (prompt_tokens, prompt_mask)
        else:
            x = batch.to(device)
            prompt = None

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, dtype=autocast_dtype):
            loss = loss_fn(model, x, prompt)
        loss.backward()
        optimizer.step()

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

        if i >= warmup_iters:
            if device == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                iter_time_ms = start_event.elapsed_time(end_event)
            else:
                peak_mem_mb = 0
                iter_time_ms = 0

            peak_memory_list.append(peak_mem_mb)
            time_list.append(iter_time_ms / 1000.0)

            print(
                f"Profile Step {i - warmup_iters + 1}/{profile_iters} - "
                f"Time: {iter_time_ms:.2f}ms, "
                f"Peak Mem: {peak_mem_mb:.2f}MB"
            )

    del x, loss
    gc.collect()
    torch.cuda.empty_cache()

    # flops must be after bc flop_counter is adding overhead to compiler
    print("Counting FLOPs...")
    flop_counter = FlopCounterMode(model, display=False)
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        x_flop = batch[0].to(device)

        x, prompt_tokens, prompt_mask = batch
        prompt_flop = (prompt_tokens, prompt_mask)
    else:
        x_flop = batch.to(device)
        prompt_flop = None

    optimizer.zero_grad(set_to_none=True)
    with flop_counter:
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            loss = loss_fn(model, x_flop, prompt_flop)
        loss.backward()

    total_flops = flop_counter.get_total_flops()
    print(f"Total FLOPs per iteration: {total_flops / 1e9:.2f} GFLOPs")

    if len(time_list) > 0:
        avg_peak_memory = np.mean(peak_memory_list)
        avg_time = np.mean(time_list)
        tflops = (total_flops / avg_time) / 1e12 if avg_time > 0 else 0

        print("\n--- Profiling Summary ---")
        print(f"Average Iteration Time: {avg_time * 1000:.2f} ms")
        print(f"Average Peak Memory: {avg_peak_memory:.2f} MB")
        print(f"Average FLOPs per Iteration: {total_flops / 1e9:.2f} GFLOPs")
        print(f"Achieved Performance: {tflops:.2f} TFLOPS")
        print("-------------------------\n")


def profile_sample_trace(
    model, schedule, config, D, device, output_dir, autocast_dtype, args
):
    flush()
    model.eval()

    warmup_iters = args.warmup_iters
    profile_iters = args.profile_iters

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    is_conditional = config.get("is_conditional", False)
    embeddings = None
    attention_mask = None
    if is_conditional:
        prompts = [""] * config.get("batch_size", 1)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                embeddings, attention_mask = model["text_enc"](prompts)

    trace_name = "sample_trace_compile" if args.compile else "sample_trace"
    with_stack = not args.compile

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0, warmup=warmup_iters, active=profile_iters, repeat=1
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
        on_trace_ready=lambda prof: trace_handler(prof, output_dir, trace_name),
    ) as prof:
        for i in range(warmup_iters + profile_iters):
            with torch.profiler.record_function("## generate_samples ##"):
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    generate_samples(
                        model=model,
                        schedule=schedule,
                        batch_size=config.get("batch_size", 1),
                        data_shape=D,
                        diffusion_type=config.get("schedule_type", "linear"),
                        prediction_target=config.get("prediction_target", "v"),
                        num_steps=args.sample_steps,
                        is_conditional=is_conditional,
                        embeddings=embeddings,
                        attention_mask=attention_mask,
                        return_traj=False,
                        device=device,
                    )
            prof.step()

    print(f"Sample Trace profiling completed. Results saved to {output_dir}")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total" if device == "cuda" else "cpu_time_total",
            row_limit=20,
        )
    )


def profile_sample_memory_flops(
    model, schedule, config, D, device, autocast_dtype, args
):
    flush()
    model.eval()

    warmup_iters = args.warmup_iters
    profile_iters = args.profile_iters

    peak_memory_list = []
    time_list = []

    is_conditional = config.get("is_conditional", False)
    embeddings = None
    attention_mask = None
    if is_conditional:
        prompts = [""] * config.get("batch_size", 1)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                embeddings, attention_mask = model["text_enc"](prompts)

    print("Counting FLOPs...")
    flop_counter = FlopCounterMode(model, display=False)
    with flop_counter:
        with torch.autocast(device_type=device, dtype=autocast_dtype):
            generate_samples(
                model=model,
                schedule=schedule,
                batch_size=config.get("batch_size", 1),
                data_shape=D,
                diffusion_type=config.get("schedule_type", "linear"),
                prediction_target=config.get("prediction_target", "v"),
                num_steps=args.sample_steps,
                is_conditional=is_conditional,
                embeddings=embeddings,
                attention_mask=attention_mask,
                return_traj=False,
                device=device,
            )
    total_flops = flop_counter.get_total_flops()
    print(f"Total FLOPs per generation: {total_flops / 1e9:.2f} GFLOPs")

    print("Benchmarking time and memory...")
    for i in range(warmup_iters + profile_iters):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        with torch.autocast(device_type=device, dtype=autocast_dtype):
            generate_samples(
                model=model,
                schedule=schedule,
                batch_size=config.get("batch_size", 1),
                data_shape=D,
                diffusion_type=config.get("schedule_type", "linear"),
                prediction_target=config.get("prediction_target", "v"),
                num_steps=args.sample_steps,
                is_conditional=is_conditional,
                embeddings=embeddings,
                attention_mask=attention_mask,
                return_traj=False,
                device=device,
            )

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

        if i >= warmup_iters:
            if device == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                iter_time_ms = start_event.elapsed_time(end_event)
            else:
                peak_mem_mb = 0
                iter_time_ms = 0

            peak_memory_list.append(peak_mem_mb)
            time_list.append(iter_time_ms / 1000.0)

            print(
                f"Sample Profile Step {i - warmup_iters + 1}/{profile_iters} - "
                f"Time: {iter_time_ms:.2f}ms, "
                f"Peak Mem: {peak_mem_mb:.2f}MB"
            )

    if len(time_list) > 0:
        avg_peak_memory = np.mean(peak_memory_list)
        avg_time = np.mean(time_list)
        tflops = (total_flops / avg_time) / 1e12 if avg_time > 0 else 0

        print("\n--- Sample Profiling Summary ---")
        print(f"Average Iteration Time: {avg_time * 1000:.2f} ms")
        print(f"Average Peak Memory: {avg_peak_memory:.2f} MB")
        print(f"Average FLOPs per Generation: {total_flops / 1e9:.2f} GFLOPs")
        print(f"Achieved Performance: {tflops:.2f} TFLOPS")
        print("-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_example.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "sample"],
        default="train",
        help="Whether to profile the training loop or the generation sampling.",
    )
    parser.add_argument(
        "--profile_type",
        type=str,
        choices=["trace", "memory_flops"],
        default="trace",
        help="Profile traces via PyTorch Profiler or measure Memory & FLOPs.",
    )
    parser.add_argument("--output_dir", type=str, default="results/profiling_results")
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--profile_iters", type=int, default=3)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=20,
        help="Number of steps when profiling sampling.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile the model before profiling."
    )
    parser.add_argument(
        "--load_into_ram",
        action="store_true",
        help="Store the images in ram",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    run_profiling(args)
