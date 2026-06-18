import torch
import torch.nn as nn
import copy
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer


class EMAModel:
    """
    Maintains an Exponential Moving Average of the model's parameters.
    Includes a context manager to seamlessly swap weights for evaluation.
    """

    def __init__(
        self, model: nn.Module = None, decay: float = 0.9999, use_ema: bool = True
    ):
        self.decay = decay
        self.use_ema = use_ema
        self.ema_model = None

        if self.use_ema and model is not None:
            self.initialize(model)

    def initialize(self, model: nn.Module):
        """Lazily initializes the EMA model weights to save memory."""
        if self.ema_model is None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
            self.use_ema = True

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.use_ema or self.ema_model is None:
            return

        # Extract uncompiled model to ensure parameter structures match perfectly
        uncompiled_model = getattr(model, "_orig_mod", model)

        for ema_param, param in zip(
            self.ema_model.parameters(), uncompiled_model.parameters()
        ):
            if param.requires_grad:
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module):
        """
        Context manager to temporarily replace the model's parameters with the EMA
        parameters for sampling/evaluation. Restores original parameters on exit.
        """
        if not self.use_ema or self.ema_model is None:
            yield
            return

        uncompiled_model = getattr(model, "_orig_mod", model)

        original_params = [p.clone().detach() for p in uncompiled_model.parameters()]

        for ema_param, param in zip(
            self.ema_model.parameters(), uncompiled_model.parameters()
        ):
            param.data.copy_(ema_param.data)

        try:
            yield
        finally:
            for orig_param, param in zip(
                original_params, uncompiled_model.parameters()
            ):
                param.data.copy_(orig_param.data)


class HFTextEncoder(nn.Module):
    """
    A general text encoder using HuggingFace's transformers library.
    Extracts the final hidden state of a causal LM to leverage implicit
    positional information inherent in causal attention patterns.
    """

    def __init__(self, model_id: str, max_seq_len: int = 256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure pad token is set for batched inference
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        # Load model in bfloat16 to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.embed_dim = self.model.config.hidden_size

    def forward(self, inputs):
        device = next(self.model.parameters()).device

        if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], str):
            encoded = self.tokenizer(
                list(inputs),
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            ).to(device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            input_ids, attention_mask = inputs
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Get the final hidden state from the last layer (before LM head)
            embeds = outputs.hidden_states[-1]

        return embeds, attention_mask


class SimpleTextEncoder(nn.Module):
    """
    A simple embedding layer that tokenizes comma-separated tags into indices
    and outputs the corresponding embeddings.
    """

    def __init__(
        self,
        vocab: dict,
        max_seq_len: int,
        embed_dim: int,
        use_pos: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)
        self.embedding = nn.Embedding(len(vocab), embed_dim, padding_idx=self.pad_id)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim) if use_pos else None

        # Initialize with small variance
        nn.init.normal_(self.embedding.weight, std=0.02)
        if use_pos:
            nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, inputs):
        device = self.embedding.weight.device

        # raw strings for inference/sampling
        if isinstance(inputs, (list, tuple)) and isinstance(inputs[0], str):
            batch_ids = []
            for prompt in inputs:
                tags = [t.strip() for t in prompt.split(",") if t.strip()]
                ids = [self.vocab.get(tag, self.unk_id) for tag in tags]
                ids = ids[: self.max_seq_len]
                batch_ids.append(ids)

            local_max_len = max((len(ids) for ids in batch_ids), default=1)

            if local_max_len <= 24:
                target_len = min(24, self.max_seq_len)
            elif local_max_len <= 52:
                target_len = min(52, self.max_seq_len)
            else:
                target_len = self.max_seq_len

            padded_batch_ids = []
            for ids in batch_ids:
                padded_ids = ids + [self.pad_id] * (target_len - len(ids))
                padded_batch_ids.append(padded_ids)

            batch_tensor = torch.tensor(
                padded_batch_ids, dtype=torch.long, device=device
            )

            not_pad_mask = batch_tensor != self.pad_id
            shifted_mask = torch.roll(not_pad_mask, shifts=1, dims=1)
            shifted_mask[:, 0] = True
            attention_mask = not_pad_mask | shifted_mask

        else:
            batch_tensor, attention_mask = inputs
            batch_tensor = batch_tensor.to(device)
            attention_mask = attention_mask.to(device)

            not_pad = batch_tensor != self.pad_id
            if not_pad.any():
                local_max_len = not_pad.sum(dim=1).max().item()
            else:
                local_max_len = 1

            if local_max_len <= 24:
                target_len = min(24, self.max_seq_len)
            elif local_max_len <= 52:
                target_len = min(52, self.max_seq_len)
            else:
                target_len = self.max_seq_len

            batch_tensor = batch_tensor[:, :target_len]
            attention_mask = attention_mask[:, :target_len]

        # Generate position IDs and add to embeddings
        seq_len = batch_tensor.size(1)
        pos_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(batch_tensor.size(0), -1)
        )

        embeds = self.embedding(batch_tensor)
        if self.pos_embedding is not None:
            embeds += self.pos_embedding(pos_ids)

        return embeds, attention_mask


def init_weights(m):
    """
    Simple weight initialization for the UNet.
    Uses Kaiming Normal for Convolutions/Linear layers to account for SiLU activations,
    and standard initialization for Normalization layers.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
        # Initialize normalization layers to be identity
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
