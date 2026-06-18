import enum
import torch
import torch.nn as nn
import math

HAS_FLASH_ATTENTION = False
try:
    from flash_attn import flash_attn_func
    from flash_attn import __version__ as fa_version

    HAS_FLASH_ATTENTION = False
    # print(f"Using flash attention with version {fa_version}")
except ImportError:
    print("Couldn't import flash attention")
    pass


_ROPE_CACHE = {}


def apply_rotary_emb(x: torch.Tensor, rotary_emb: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embeddings to the input tensor.
    Args:
        x: Tensor of shape [B, SeqLen, Heads, HeadDim]
        rotary_emb: Tuple of (cos, sin) or a single tensor containing both.
                    Shape should be broadcastable to [B, SeqLen, 1, HeadDim]
    """
    # Assuming rotary_emb is concatenated [cos, sin] on the last dimension
    cos, sin = rotary_emb.chunk(2, dim=-1)

    # Expand to match the Heads dimension: [B, SeqLen, 1, HeadDim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    # Rotate half the hidden dims
    x1, x2 = x.chunk(2, dim=-1)
    x_rot = torch.cat([-x2, x1], dim=-1)

    return x * cos + x_rot * sin


def get_2d_rotary_pos_embed(embed_dim, h, w, device):
    """Generates 2D Rotary Position Embeddings and caches them."""
    key = (embed_dim, h, w, device)
    if key in _ROPE_CACHE:
        return _ROPE_CACHE[key]

    dim_half = embed_dim // 2
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, dim_half, 2, device=device).float() / dim_half)
    )
    y = torch.arange(h, device=device, dtype=torch.float32)
    x = torch.arange(w, device=device, dtype=torch.float32)

    freqs_y = torch.outer(y, inv_freq)
    freqs_x = torch.outer(x, inv_freq)

    freqs_y = freqs_y.repeat_interleave(2, dim=-1)
    freqs_x = freqs_x.repeat_interleave(2, dim=-1)

    freqs_y = freqs_y.view(h, 1, -1).repeat(1, w, 1)
    freqs_x = freqs_x.view(1, w, -1).repeat(h, 1, 1)

    freqs = torch.cat([freqs_y, freqs_x], dim=-1).view(h * w, embed_dim)
    emb = torch.cat([freqs.cos(), freqs.sin()], dim=-1).unsqueeze(
        0
    )  # [1, SeqLen, HeadDim * 2]

    _ROPE_CACHE[key] = emb
    return emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        # scale t=1.0 input
        if x.max() <= 1.001:
            x = x * 1000.0

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeEmbeddings(nn.Module):
    """
    Calculates sinusoidal embeddings and projects them.
    Matches diffusers Timesteps + TimestepEmbedding structure.
    """

    def __init__(self, sinusoidal_dim: int, output_dim: int, max_period=10000):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.output_dim = output_dim
        if sinusoidal_dim % 2 != 0:
            raise ValueError(
                f"Cannot use sinusoidal dim {sinusoidal_dim}, must be even."
            )
        half_dim = sinusoidal_dim // 2

        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device="cuda"
        )
        exponent = exponent / half_dim

        self.register_buffer("inv_freq", torch.exp(exponent), persistent=False)

        self.linear_1 = nn.Linear(sinusoidal_dim, output_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def _get_sinusoidal_embeddings(self, timesteps: torch.Tensor):
        """Calculates the base sinusoidal embeddings."""
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        emb = timesteps[:, None].float() * self.inv_freq[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        half_dim = self.sinusoidal_dim // 2
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.sinusoidal_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb

    def forward(self, timesteps: torch.Tensor, sample: torch.Tensor):
        if timesteps.max() <= 1.001:
            timesteps = timesteps * 1000.0

        timesteps = timesteps.expand(sample.shape[0])
        sin_emb = self._get_sinusoidal_embeddings(timesteps)

        sin_emb = sin_emb.to(dtype=self.linear_1.weight.dtype)

        emb = self.linear_1(sin_emb)
        emb = self.act(emb)
        emb = self.linear_2(emb)
        return emb


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embeddings_channels: int,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings_channels = time_embeddings_channels
        self.norm_num_groups = norm_num_groups
        self.eps = eps

        self.norm1 = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            num_channels=self.in_channels,
            eps=self.eps,
        )
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        # applied after conv1 so use out_channels
        if time_embeddings_channels is not None:
            self.time_emb_proj = nn.Linear(
                in_features=time_embeddings_channels,
                out_features=self.out_channels,
                bias=True,
            )
        self.nonlinearity = nn.SiLU()

        self.norm2 = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            num_channels=self.out_channels,
            eps=self.eps,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        use_conv_shortcut = True if self.in_channels != self.out_channels else False
        self.conv_shortcut = None
        if use_conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, x, temb):
        hidden_states = x

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Diffusers: temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        output = hidden_states + x

        return output


class EfficientResnetBlock2D(nn.Module):
    """
    Replaces standard convolutions with Expanded Separable Convolutions (UIB block-like).
    DW -> PW (Expand) -> Norm -> Act -> DW -> PW (Reduce)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embeddings_channels: int,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
        expansion_ratio: float = 2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings_channels = time_embeddings_channels
        self.norm_num_groups = norm_num_groups
        self.eps = eps

        hidden_channels = int(out_channels * expansion_ratio)

        self.norm1 = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=eps
        )
        # dw1 processes the input channels independently
        self.dw1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        # pw1 handles the projection from the input to the expanded hidden dimension
        self.pw1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)

        if time_embeddings_channels is not None:
            self.time_emb_proj = nn.Linear(
                time_embeddings_channels, hidden_channels, bias=True
            )

        self.norm2 = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=hidden_channels, eps=eps
        )
        self.dropout = nn.Dropout(p=dropout)
        self.dw2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_channels,
            bias=False,
        )
        self.pw2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
            )

    def forward(self, x, temb):
        res = x

        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.dw1(h)
        h = self.pw1(h)

        if temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]
            h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.dw2(h)
        h = self.pw2(h)

        if self.conv_shortcut is not None:
            res = self.conv_shortcut(res)

        return h + res


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.scale_factor = 2.0

        self.conv = None
        if use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )

    def forward(self, x):
        # interpolate fails for big tensors

        if x.shape[0] >= 64 or x.numel() * self.scale_factor > pow(2, 31):
            x = x.contiguous()
        x = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )
        if self.conv is not None:
            x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        in_channels,
        cross_attention_dim=None,
        num_attention_heads=8,
        qk_norm=None,
        kv_num_heads=None,
        eps: float = 1e-5,
        base_sequence_length: int = None,
        use_flash_attention=HAS_FLASH_ATTENTION,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attention_dim = in_channels
        self.num_attention_heads = num_attention_heads
        if cross_attention_dim:
            self.cross_attention_dim = cross_attention_dim

        self.use_flash_attention = use_flash_attention
        self.base_sequence_length = base_sequence_length
        self.head_dim = self.in_channels // self.num_attention_heads

        # If MQA (kv_num_heads=1), kv_dim is exactly 1 * head_dim (e.g., 64)
        self.kv_num_heads = (
            kv_num_heads if kv_num_heads is not None else self.num_attention_heads
        )
        self.kv_dim = self.head_dim * self.kv_num_heads

        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(
            in_features=self.in_channels, out_features=self.in_channels, bias=False
        )
        self.to_k = nn.Linear(
            in_features=self.cross_attention_dim,
            out_features=self.kv_dim,
            bias=False,
        )
        self.to_v = nn.Linear(
            in_features=self.cross_attention_dim,
            out_features=self.kv_dim,
            bias=False,
        )

        self.to_out = nn.Linear(
            in_features=self.in_channels, out_features=self.in_channels, bias=False
        )

        self.norm_q = None
        self.norm_k = None
        if qk_norm == "rms_norm":
            self.norm_q = nn.RMSNorm(
                self.head_dim,
                eps=eps,
                elementwise_affine=True,
            )
            self.norm_k = nn.RMSNorm(
                self.head_dim,
                eps=eps,
                elementwise_affine=True,
            )

        if self.use_flash_attention:
            self.forward = self.forward_flash_attention
        else:
            self.forward = self.forward_sdpa

    def forward_sdpa(
        self, x, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        batch, T, C = x.shape
        q = self.to_q(x)

        encoder_hidden_states = (
            encoder_hidden_states if encoder_hidden_states is not None else x
        )
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # reshape with multi heads
        # GQA: q[B, H*W, C] -> [B, H*W, Heads, C/Heads], k: [B, T, C] -> [B, H*W, KV_Heads, C/KV_Heads]
        q = q.view(batch, -1, self.num_attention_heads, self.head_dim)
        k = k.view(batch, -1, self.kv_num_heads, self.head_dim)
        v = v.view(batch, -1, self.kv_num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if attention_mask is not None:
            # attention_mask is [B, T_k]
            # SDPA expects mask of shape [B, 1, 1, T_k] to broadcast over heads and queries
            sdpa_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            sdpa_mask = None

        # Apply RoPE after QK-norm
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        # Transpose for SDPA: [B, Heads, SeqLen, HeadDim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # repeat KV heads to match Q heads
        if self.num_attention_heads != self.kv_num_heads:
            n_rep = self.num_attention_heads // self.kv_num_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        softmax_scale = self.scale
        if self.base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(T, self.base_sequence_length)) * self.scale
            )

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, is_causal=False, scale=softmax_scale
        )

        # attn: [B, Heads, H*W, C/Heads] -> [B, H*W, Heads, C/Heads]
        x = x.transpose(1, 2).reshape(batch, -1, C)

        x = self.to_out(x)
        return x

    def forward_flash_attention(
        self, x, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None
    ):
        # the input is [B, T, C]
        B, T, C = x.shape
        q = self.to_q(x)

        encoder_hidden_states = (
            encoder_hidden_states if encoder_hidden_states is not None else x
        )
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # q [B, H*W, Heads, HeadDim], k [B, T, KV_Heads, HeadDim]
        q = q.view(B, -1, self.num_attention_heads, self.head_dim)
        k = k.view(B, -1, self.kv_num_heads, self.head_dim)
        v = v.view(B, -1, self.kv_num_heads, self.head_dim)

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

        softmax_scale = self.scale
        if self.base_sequence_length is not None:
            softmax_scale = (
                math.sqrt(math.log(T, self.base_sequence_length)) * self.scale
            )

        x = flash_attn_func(q, k, v, causal=False, softmax_scale=softmax_scale)

        # attn [B, H*W, NH, NDIM]
        x = x.contiguous().view(B, -1, C)

        x = self.to_out(x)

        return x


class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        # concatenate output dims
        self.proj_in = nn.Linear(
            in_features=self.in_channels,
            out_features=self.out_channels * 2,
            bias=self.bias,
        )

    def forward(self, x):
        hidden_states, gate = self.proj_in(x).chunk(2, dim=-1)
        return hidden_states * torch.nn.functional.gelu(gate)


class Feedforward(nn.Module):
    def __init__(self, in_channels, expansion_ratio=4) -> None:
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = int(in_channels * expansion_ratio)

        self.geglu = GEGLU(self.in_channels, hidden_channels, bias=True)
        self.proj_out = nn.Linear(
            in_features=hidden_channels, out_features=self.in_channels, bias=True
        )

    def forward(self, x):
        x = self.geglu(x)
        x = self.proj_out(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        cross_attention_dim=None,
        num_attention_heads: int = 8,
        use_checkpointing: bool = True,
        disable_self_attention: bool = False,
        qk_norm: str = None,
        kv_num_heads: int = None,
        ffn_expansion_ratio: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cross_attention_dim = cross_attention_dim
        self.num_attention_heads = num_attention_heads
        self.use_checkpointing = use_checkpointing
        self.disable_self_attention = disable_self_attention

        # self attn
        if not self.disable_self_attention:
            self.norm1 = nn.LayerNorm(self.in_channels)
            self.attn1 = Attention(
                in_channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                qk_norm=qk_norm,
                kv_num_heads=kv_num_heads,
            )

        # cross attn
        if cross_attention_dim is not None:
            self.norm2 = nn.LayerNorm(self.in_channels)
            self.attn2 = Attention(
                in_channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                cross_attention_dim=self.cross_attention_dim,
                qk_norm=qk_norm,
                kv_num_heads=kv_num_heads,
            )

        # feed forward
        self.norm3 = nn.LayerNorm(self.in_channels)
        self.ff = Feedforward(self.in_channels, expansion_ratio=ffn_expansion_ratio)

    def forward(
        self, x, encoder_hidden_states, attention_mask=None, image_rotary_emb=None
    ):
        if not self.disable_self_attention:
            x_norm = self.norm1(x)
            # attention mask if there is NO cross-attention (imgs no mask)
            self_attn_mask = attention_mask if encoder_hidden_states is None else None
            hidden_states = x + self.attn1(
                x_norm, attention_mask=self_attn_mask, image_rotary_emb=image_rotary_emb
            )
        else:
            hidden_states = x

        if encoder_hidden_states is not None:
            hidden_states_norm = self.norm2(hidden_states)
            hidden_states = hidden_states + self.attn2(
                hidden_states_norm, encoder_hidden_states, attention_mask=attention_mask
            )

        hidden_states_norm = self.norm3(hidden_states)
        if self.use_checkpointing:
            ff_out = torch.utils.checkpoint.checkpoint(
                self.ff, hidden_states_norm, use_reentrant=True
            )
            hidden_states = hidden_states + ff_out
        else:
            hidden_states = hidden_states + self.ff(hidden_states_norm)

        return hidden_states


class TransformerTextAdapter(nn.Module):
    """
    A strong transformer-based text adapter as proposed in the i1 paper.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        ffn_expansion_ratio: float = 4.0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    in_channels=hidden_size,
                    cross_attention_dim=None,  # Self-attention only
                    num_attention_heads=num_attention_heads,
                    use_checkpointing=False,
                    disable_self_attention=False,
                    qk_norm="rms_norm",
                    ffn_expansion_ratio=ffn_expansion_ratio,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attention_mask=None):
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, encoder_hidden_states=None, attention_mask=attention_mask)
        return x


class Transformer2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cross_attention_dim,
        num_attention_heads,
        num_layers: int = 1,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        use_checkpointing: bool = True,
        disable_self_attention: bool = False,
        use_rope: bool = False,
        qk_norm: str = None,
        kv_num_heads: int = None,
        ffn_expansion_ratio: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.norm_num_groups = norm_num_groups
        self.eps = eps
        self.use_rope = use_rope

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=self.in_channels, eps=self.eps
        )
        self.proj_in = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    in_channels,
                    cross_attention_dim,
                    num_attention_heads,
                    use_checkpointing=use_checkpointing,
                    disable_self_attention=disable_self_attention,
                    qk_norm=qk_norm,
                    kv_num_heads=kv_num_heads,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.proj_out = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x, encoder_hidden_states, attention_mask=None):
        batch, _, height, width = x.shape
        res = x

        x = self.norm(x)
        x = self.proj_in(x)
        inner_dim = x.shape[1]
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)

        image_rotary_emb = None
        if self.use_rope:
            head_dim = inner_dim // self.num_attention_heads
            image_rotary_emb = get_2d_rotary_pos_embed(
                head_dim, height, width, x.device
            )

        for block in self.transformer_blocks:
            x = block(
                x,
                encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        # prepare output
        x = x.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)

        x = x + res

        return x
