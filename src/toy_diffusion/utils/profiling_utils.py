import functools
import gc
import logging
import os
import torch
import torch.profiler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def annotate(func, name):
    """Wrap a function with torch.profiler.record_function for trace annotation."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(name):
            return func(*args, **kwargs)

    return wrapper


def annotate_model(model):
    """Apply profiler annotations to key model methods."""
    if isinstance(model, (dict, torch.nn.ModuleDict)):
        if "unet" in model:
            func = getattr(model["unet"].forward, "__func__", model["unet"].forward)
            model["unet"].forward = annotate(func, "unet_forward").__get__(
                model["unet"], type(model["unet"])
            )
        if "text_enc" in model:
            func = getattr(
                model["text_enc"].forward, "__func__", model["text_enc"].forward
            )
            model["text_enc"].forward = annotate(func, "text_enc_forward").__get__(
                model["text_enc"], type(model["text_enc"])
            )
    else:
        func = getattr(model.forward, "__func__", model.forward)
        model.forward = annotate(func, "unet_forward").__get__(model, type(model))


def flush():
    """Flushes memory to avoid OOMs and reset tracking stats."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()


def trace_handler(prof: torch.profiler.profile, output_dir: str, name: str):
    """Exports Chrome traces and Memory timelines."""
    os.makedirs(output_dir, exist_ok=True)
    trace_file = os.path.join(output_dir, f"{name}_trace.json.gz")
    prof.export_chrome_trace(trace_file)
    logger.info(f"Chrome trace saved to: {trace_file}")

    memory_file = os.path.join(output_dir, f"{name}_memory.html")
    try:
        prof.export_memory_timeline(memory_file, device="cuda:0")
        logger.info(f"Memory timeline saved to: {memory_file}")
    except Exception as e:
        logger.warning(f"Could not export memory timeline: {e}")

