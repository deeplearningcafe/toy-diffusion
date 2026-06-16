import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_diffusion.models.layers import TimeEmbeddings


# based on https://github.com/zlab-princeton/i1/blob/main/torch_inference/generate.py
def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    """Splits the head dimension into 3 chunks for Text, Y, and X coordinates."""
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    dtype = x.dtype
    # x shape: [B, SeqLen, Heads, HeadDim]
    x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)

    # cos, sin shape: [B, SeqLen, HeadDim // 2] -> [B, SeqLen, 1, HeadDim // 2]
    cos = cos.unsqueeze(2).float()
    sin = sin.unsqueeze(2).float()

    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape_as(x).to(dtype)


class MultimodalRopeEmbedder(nn.Module):
    """3D RoPE for Joint Text and Image streams."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_text_len: int = 512,
        max_spatial_dim: int = 128,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        axes_lens = (max_text_len, max_spatial_dim, max_spatial_dim)

        cos_tables = []
        sin_tables = []
        for dim, axis_len in zip(axes_dims, axes_lens):
            steps = torch.arange(0, dim, 2, dtype=torch.float32)
            base = 1.0 / (theta ** (steps / dim))
            positions = torch.arange(axis_len, dtype=torch.float32)
            angles = positions[:, None] * base[None, :]
            cos_tables.append(angles.cos())
            sin_tables.append(angles.sin())

        self.cos_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in cos_tables]
        )
        self.sin_tables = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in sin_tables]
        )

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = []
        sin = []
        for axis_idx, (cos_table, sin_table) in enumerate(
            zip(self.cos_tables, self.sin_tables)
        ):
            pos = position_ids[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(F.embedding(pos, cos_table))
            sin.append(F.embedding(pos, sin_table))
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MMDiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)

        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: tuple[torch.Tensor, torch.Tensor],
        text_freqs: tuple[torch.Tensor, torch.Tensor],
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(linear: nn.Linear, x: torch.Tensor):
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q, k, v

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)

        q_image, k_image = self.q_norm(q_image), self.k_norm(k_image)
        q_text, k_text = self.q_norm(q_text), self.k_norm(k_text)

        # Apply 3D RoPE
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)

        # Transpose for SDPA: [B, Heads, SeqLen, HeadDim]
        q = torch.cat([q_image, q_text], dim=1).transpose(1, 2)
        k = torch.cat([k_image, k_text], dim=1).transpose(1, 2)
        v = torch.cat([v_image, v_text], dim=1).transpose(1, 2)

        image_mask = torch.ones(
            (bsz, image_len), dtype=torch.bool, device=text_tokens.device
        )
        key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
        attn_mask = key_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(bsz, image_len + text_len, self.hidden_size)

        # Zero out masked text tokens
        out = out * key_mask[:, :, None].to(out.dtype)

        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class DualStreamDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        eps: float = 1e-5,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)

        # Sandwich Norm
        self.norm1 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm3 = nn.RMSNorm(hidden_size, eps=eps)
        self.norm4 = nn.RMSNorm(hidden_size, eps=eps)

        self.attn = MMDiTAttention(hidden_size, num_heads, eps=eps)

        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio))
        self.mlp_image = SwiGLUFFN(hidden_size, hidden_features)
        self.mlp_text = SwiGLUFFN(hidden_size, hidden_features)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: tuple[torch.Tensor, torch.Tensor],
        text_freqs: tuple[torch.Tensor, torch.Tensor],
        text_mask: torch.Tensor,
        skip: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.use_skip and skip is not None:
            image_tokens = self.skip_linear_image(
                torch.cat([image_tokens, skip[0]], dim=-1)
            )
            text_tokens = self.skip_linear_text(
                torch.cat([text_tokens, skip[1]], dim=-1)
            )

        image_attn, text_attn = self.attn(
            self.norm1(image_tokens),
            self.norm1(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )

        image_tokens = image_tokens + self.norm3(image_attn)
        text_tokens = text_tokens + self.norm3(text_attn)

        image_tokens = image_tokens + self.norm4(
            self.mlp_image(self.norm2(image_tokens))
        )
        text_tokens = text_tokens + self.norm4(self.mlp_text(self.norm2(text_tokens)))

        text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class DualStreamDiT(nn.Module):
    """
    Lightweight Dual-Stream DiT based on the i1 paper.
    Uses Time Token prepending instead of AdaLN to support Flow Matching efficiently.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 768,
        use_checkpointing: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_checkpointing = use_checkpointing

        # 1. Image Embedder
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # 2. Time Embedder (Used as a prepended token)
        self.time_embedding = TimeEmbeddings(sinusoidal_dim=256, output_dim=hidden_size)
        self.time_token_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

        # 3. Text Adapter
        self.text_adapter = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # 4. 3D RoPE
        head_dim = hidden_size // num_heads
        axes_dims = _default_rope_axes_dims(head_dim)
        self.rope_embedder = MultimodalRopeEmbedder(axes_dims)

        # 5. Dual Stream Blocks (with Long Skip Connections)
        num_in_blocks = depth // 2
        self.in_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(hidden_size, num_heads, mlp_ratio, eps=eps)
                for _ in range(num_in_blocks)
            ]
        )

        self.mid_block = DualStreamDiTBlock(hidden_size, num_heads, mlp_ratio, eps=eps)

        self.out_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size, num_heads, mlp_ratio, eps=eps, use_skip=True
                )
                for _ in range(num_in_blocks)
            ]
        )

        self.norm_final = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self._zero_initialize_output()

    def _zero_initialize_output(self):
        """Crucial for diffusion/flow matching: start by predicting zero velocity/noise."""
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _build_position_ids(
        self, text_mask: torch.Tensor, h: int, w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, text_len = text_mask.shape
        device = text_mask.device

        # Text Coordinates: (pos, 0, 0). Time token is at pos 0, words are 1..text_len
        caption_positions = torch.arange(text_len, dtype=torch.long, device=device)[
            None
        ].expand(bsz, text_len)
        caption_positions = torch.where(
            text_mask.bool(), caption_positions, torch.zeros_like(caption_positions)
        )
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)

        # Image Coordinates: (L, y, x). L is the length of the valid text prompt.
        num_image_tokens = h * w
        text_lengths = text_mask.sum(dim=1, dtype=torch.long)

        row_ids = (
            torch.arange(h, device=device)
            .repeat_interleave(w)[None]
            .expand(bsz, num_image_tokens)
        )
        col_ids = (
            torch.arange(w, device=device).repeat(h)[None].expand(bsz, num_image_tokens)
        )
        image_time = text_lengths[:, None].expand(bsz, num_image_tokens)

        image_ids = torch.stack((image_time, row_ids, col_ids), dim=-1)

        return caption_ids, image_ids

    def _checkpoint(self, module, *args, **kwargs):
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                module, *args, **kwargs, use_reentrant=False
            )
        else:
            return module(*args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, H, W = x.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        # 1. Patchify Image
        image_tokens = self.x_embedder(x).flatten(2).transpose(1, 2)  # [B, H*W, C]

        # 2. Prepare Time Token
        t_emb = self.time_embedding(t, x)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)  # [B, 1, C]

        # 3. Prepare Text Tokens
        text_tokens = self.text_adapter(encoder_hidden_states)

        text_tokens = torch.cat([time_token, text_tokens], dim=1)

        time_mask = torch.ones((bsz, 1), dtype=torch.bool, device=x.device)
        text_mask = torch.cat([time_mask, attention_mask.bool()], dim=1)

        # 3D RoPE Frequencies
        text_pos_ids, image_pos_ids = self._build_position_ids(
            text_mask, h_patches, w_patches
        )
        all_pos_ids = torch.cat([text_pos_ids, image_pos_ids], dim=1)
        cos, sin = self.rope_embedder(all_pos_ids)

        seq_text = text_tokens.shape[1]
        text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
        image_freqs = (cos[:, seq_text:], sin[:, seq_text:])

        skips = []
        for block in self.in_blocks:
            image_tokens, text_tokens = self._checkpoint(
                block, image_tokens, text_tokens, image_freqs, text_freqs, text_mask
            )
            skips.append((image_tokens, text_tokens))

        image_tokens, text_tokens = self._checkpoint(
            self.mid_block,
            image_tokens,
            text_tokens,
            image_freqs,
            text_freqs,
            text_mask,
        )

        for block in self.out_blocks:
            skip_tensors = skips.pop()
            image_tokens, text_tokens = self._checkpoint(
                block,
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
                skip=skip_tensors,
            )

        tokens = self.proj_out(self.norm_final(image_tokens))  # [B, H*W, p*p*C_out]

        tokens = tokens.reshape(bsz, h_patches, w_patches, p, p, self.out_channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.out_channels, H, W)

        return tokens
