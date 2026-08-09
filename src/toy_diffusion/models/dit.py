import torch
import torch.nn as nn
from liger_kernel.transformers import LigerRMSNorm

from toy_diffusion.models.layers import (
    Attention,
    Feedforward,
    TimeEmbeddings,
    TransformerTextAdapter,
    MultimodalRopeEmbedder,
    _default_rope_axes_dims,
)


class LuminaRMSNormZero(nn.Module):
    """
    Adaptive RMS normalization zero.
    Returns the normalized tensor alongside the un-applied gates and scales
    to be used in the Sandwich Normalization architecture.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            embedding_dim,
            4 * embedding_dim,
            bias=True,
        )
        self.norm = LigerRMSNorm(embedding_dim, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None, :])

        return x, gate_msa, scale_mlp, gate_mlp


class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock implementing Sandwich Normalization.
    Supports original AdaLN modulation and unmodulated (i1) mode,
    along with gradient checkpointing outside attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        eps: float = 1e-5,
        base_sequence_length: int = 256,
        use_i1: bool = False,
        use_skip: bool = False,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_i1 = use_i1
        self.use_skip = use_skip
        self.use_checkpointing = use_checkpointing

        if use_skip:
            self.skip_linear = nn.Linear(2 * dim, dim)

        if not use_i1:
            self.norm1 = LuminaRMSNormZero(embedding_dim=dim, eps=eps)
        else:
            self.norm1 = LigerRMSNorm(dim, eps=eps)

        self.attn1 = Attention(
            in_channels=dim,
            num_attention_heads=num_attention_heads,
            kv_num_heads=num_kv_heads,
            qk_norm="rms_norm",
            eps=eps,
            base_sequence_length=base_sequence_length,
        )

        self.norm2 = LigerRMSNorm(dim, eps=eps)

        self.ffn_norm1 = LigerRMSNorm(dim, eps=eps)
        self.feed_forward = Feedforward(in_channels=dim)
        self.ffn_norm2 = LigerRMSNorm(dim, eps=eps)

    def _checkpoint(self, module, *args, **kwargs):
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                module, *args, **kwargs, use_reentrant=False
            )
        else:
            return module(*args, **kwargs)

    def _run_skip(
        self,
        hidden_states: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """Encapsulates skip connection calculations for checkpointing."""
        return self.skip_linear(torch.cat([hidden_states, skip], dim=-1))

    def _run_post_attn_i1(
        self,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
    ) -> torch.Tensor:
        """Groups i1 post-attention calculations to discard intermediates."""
        hidden_states = residual + self.norm2(attn_output)

        mlp_input = self.ffn_norm1(hidden_states)
        mlp_output = self.feed_forward(mlp_input)
        return hidden_states + self.ffn_norm2(mlp_output)

    def _run_post_attn_adaln(
        self,
        residual: torch.Tensor,
        attn_output: torch.Tensor,
        gate_msa: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """Groups AdaLN post-attention calculations for checkpointing."""
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(
            attn_output
        )

        mlp_input = self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
        mlp_output = self.feed_forward(mlp_input)
        return hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        skip: torch.Tensor = None,
    ):
        if self.use_skip and skip is not None:
            hidden_states = self._checkpoint(self._run_skip, hidden_states, skip)

        residual = hidden_states

        if not self.use_i1:
            # AdaLN modulation
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
                hidden_states, temb
            )
            attn_output = self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = self._checkpoint(
                self._run_post_attn_adaln,
                residual,
                attn_output,
                gate_msa,
                scale_mlp,
                gate_mlp,
            )
        else:
            # Unmodulated Sandwich Normalization
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = self._checkpoint(
                self._run_post_attn_i1,
                residual,
                attn_output,
            )

        return hidden_states


class LuminaNextDit(nn.Module):
    """
    Lumina Next-DiT Architecture.
    Implements a single-stream diffusion transformer. Supports original
    Lumina 2.0 mode with AdaLN and the improved i1 recipe (no AdaLN,
    time token prepending, modality refiners, 3D RoPE, and long skips).
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        out_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,
        cross_attention_dim: int = 1024,
        base_sequence_length: int = 256,
        eps: float = 1e-5,
        use_i1: bool = False,
        use_skip: bool = False,
        use_checkpointing: bool = True,
        use_rope_text_adapter: bool = False,
        norm_type: str = "layer_norm",
        activation_func: str = "geglu",
        skip_checkpointing_layers: int = 0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.use_i1 = use_i1
        self.use_skip = use_skip
        self.use_checkpointing = use_checkpointing
        self.skip_checkpointing_layers = skip_checkpointing_layers
        # in the original i1 paper the don't use it
        self.use_rope_text_adapter = use_rope_text_adapter

        def should_checkpoint(layer_idx: int) -> bool:
            return self.use_checkpointing and (layer_idx >= self.skip_checkpointing_layers)

        current_layer_idx = 0

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=256,
            output_dim=hidden_size,
        )

        if self.use_i1:
            self.time_token_proj = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, hidden_size)
            )

            text_adapter_layers = 2
            self.text_adapter = TransformerTextAdapter(
                in_channels=cross_attention_dim,
                hidden_size=hidden_size,
                num_layers=text_adapter_layers,
                num_attention_heads=num_attention_heads,
                ffn_expansion_ratio=4.0,
                use_checkpointing=should_checkpoint(current_layer_idx),
                norm_type=norm_type,
                activation_func=activation_func,
            )
            current_layer_idx += text_adapter_layers

            image_adapter_layers = 2
            self.image_refiner = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        dim=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        eps=eps,
                        base_sequence_length=None,
                        use_i1=True,
                        use_checkpointing=should_checkpoint(current_layer_idx),
                    )
                    for _ in range(image_adapter_layers)
                ]
            )
            current_layer_idx += image_adapter_layers

            head_dim = hidden_size // num_attention_heads
            axes_dims = _default_rope_axes_dims(head_dim)
            self.rope_embedder = MultimodalRopeEmbedder(axes_dims)
        else:
            self.cap_embedder = nn.Linear(cross_attention_dim, hidden_size)

        if self.use_skip:
            num_in = depth // 2
            in_start_idx = current_layer_idx
            self.in_blocks = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        dim=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        eps=eps,
                        base_sequence_length=None,
                        use_i1=use_i1,
                        use_checkpointing=should_checkpoint(in_start_idx+1),
                    )
                    for i in range(num_in)
                ]
            )
            current_layer_idx += num_in

            self.mid_block = LuminaNextDiTBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                eps=eps,
                base_sequence_length=None,
                use_i1=use_i1,
                use_checkpointing=should_checkpoint(current_layer_idx),
            )
            current_layer_idx += 1

            self.out_blocks = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        dim=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        eps=eps,
                        base_sequence_length=None,
                        use_i1=use_i1,
                        use_skip=True,
                        use_checkpointing=should_checkpoint(in_start_idx+1),
                    )
                    for i in range(num_in)
                ]
            )
            current_layer_idx += num_in
        else:
            self.blocks = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        dim=hidden_size,
                        num_attention_heads=num_attention_heads,
                        num_kv_heads=num_kv_heads,
                        eps=eps,
                        base_sequence_length=base_sequence_length,
                        use_i1=use_i1,
                        use_checkpointing=use_checkpointing,
                    )
                    for _ in range(depth)
                ]
            )

        self.norm_out = LigerRMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        self._zero_initialize_output()

    def get_params(self,) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _zero_initialize_output(self):
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        return x, (H, W)

    def unpatchify(self, x, H, W):
        B, _, _ = x.shape
        p = self.patch_size
        x = x.view(B, H // p, W // p, p, p, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return x

    def _build_position_ids(
        self, text_mask: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        bsz, text_len = text_mask.shape
        device = text_mask.device

        caption_positions = torch.arange(text_len, dtype=torch.long, device=device)[
            None
        ].expand(bsz, text_len)
        caption_positions = torch.where(
            text_mask.bool(),
            caption_positions,
            torch.zeros_like(caption_positions),
        )
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)

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

        return torch.cat([caption_ids, image_ids], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
        bsz, _, H, W = x.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        x_patches, (H, W) = self.patchify(x)
        img_tokens = self.x_embedder(x_patches)

        t_emb = self.time_embedding(t, x)

        if self.use_i1:
            # i1 Pathway
            time_token = self.time_token_proj(t_emb).unsqueeze(1)

            if attention_mask is not None:
                time_mask = torch.ones((bsz, 1), dtype=torch.bool, device=x.device)
                full_text_mask = torch.cat([time_mask, attention_mask.bool()], dim=1)
            else:
                full_text_mask = torch.ones(
                    (bsz, encoder_hidden_states.shape[1]),
                    dtype=torch.bool,
                    device=x.device,
                )

            all_pos_ids = self._build_position_ids(full_text_mask, h_patches, w_patches)
            cos, sin = self.rope_embedder(all_pos_ids)

            # Expand cos and sin to full HeadDim for apply_rotary_emb
            cos_full = torch.cat([cos, cos], dim=-1)
            sin_full = torch.cat([sin, sin], dim=-1)
            rotary_emb = torch.cat([cos_full, sin_full], dim=-1)

            # Extract Text and Image RoPE slices
            text_len = encoder_hidden_states.shape[1]
            text_rotary_emb = None
            if self.use_rope_text_adapter:
                text_rotary_emb = rotary_emb[:, 1 : 1 + text_len]
            img_rotary_emb = rotary_emb[:, 1 + text_len :]

            text_tokens = self.text_adapter(
                encoder_hidden_states,
                attention_mask=attention_mask,
                text_rotary_emb=text_rotary_emb,
            )
            text_tokens = torch.cat([time_token, text_tokens], dim=1)

            for block in self.image_refiner:
                img_tokens = block(img_tokens, image_rotary_emb=img_rotary_emb)

            full_sequence = torch.cat([text_tokens, img_tokens], dim=1)

            img_mask = torch.ones(
                (bsz, img_tokens.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
            full_mask = torch.cat([full_text_mask, img_mask], dim=1)

            hidden_states = full_sequence
            if self.use_skip:
                skips = []
                for block in self.in_blocks:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=full_mask,
                        image_rotary_emb=rotary_emb,
                    )
                    skips.append(hidden_states)

                hidden_states = self.mid_block(
                    hidden_states,
                    attention_mask=full_mask,
                    image_rotary_emb=rotary_emb,
                )

                for block in self.out_blocks:
                    skip_tensor = skips.pop()
                    hidden_states = block(
                        hidden_states,
                        attention_mask=full_mask,
                        image_rotary_emb=rotary_emb,
                        skip=skip_tensor,
                    )
            else:
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=full_mask,
                        image_rotary_emb=rotary_emb,
                    )

            img_len = img_tokens.shape[1]
            hidden_states = hidden_states[:, -img_len:]

        else:
            # Lumina NextDiT
            if encoder_hidden_states is not None:
                text_tokens = self.cap_embedder(encoder_hidden_states)
                hidden_states = torch.cat([text_tokens, img_tokens], dim=1)

                if attention_mask is not None:
                    img_mask = torch.ones(
                        (bsz, img_tokens.shape[1]),
                        dtype=torch.bool,
                        device=x.device,
                    )
                    full_mask = torch.cat([attention_mask.bool(), img_mask], dim=1)
                else:
                    full_mask = None
            else:
                hidden_states = img_tokens
                full_mask = None

            if self.use_skip:
                skips = []
                for block in self.in_blocks:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=t_emb,
                        attention_mask=full_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    skips.append(hidden_states)

                hidden_states = self.mid_block(
                    hidden_states=hidden_states,
                    temb=t_emb,
                    attention_mask=full_mask,
                    image_rotary_emb=image_rotary_emb,
                )

                for block in self.out_blocks:
                    skip_tensor = skips.pop()
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=t_emb,
                        attention_mask=full_mask,
                        image_rotary_emb=image_rotary_emb,
                        skip=skip_tensor,
                    )
            else:
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=t_emb,
                        attention_mask=full_mask,
                        image_rotary_emb=image_rotary_emb,
                    )

            if encoder_hidden_states is not None:
                img_len = img_tokens.shape[1]
                hidden_states = hidden_states[:, -img_len:]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        x = self.unpatchify(hidden_states, H, W)

        return x
