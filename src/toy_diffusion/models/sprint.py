import torch
import torch.nn as nn
from toy_diffusion.models.dit import LuminaNextDit, LuminaNextDiTBlock
from toy_diffusion.models.layers import (
    TimeEmbeddings,
    TransformerTextAdapter,
)
from toy_diffusion.models.dual_stream import (
    DualStreamDiT,
    DualStreamDiTBlock,
    MultimodalRopeEmbedder,
    _default_rope_axes_dims,
)


def random_token_drop(x, drop_ratio):
    """
    Randomly drops tokens from the sequence according to the specified ratio.
    """
    B, N, D = x.shape
    K = int(round(N * (1 - drop_ratio)))
    K = max(1, min(K, N - 1))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :K]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return x_masked, ids_keep, ids_restore


def restore_full_sequence(x_masked, ids_restore, mask_token):
    """
    Reconstructs the full sequence by padding dropped indices with mask tokens.
    """
    B, K, D = x_masked.shape
    N = ids_restore.shape[1]

    mask_tokens = mask_token.expand(B, N - K, D)
    x_concat = torch.cat([x_masked, mask_tokens], dim=1)

    index = ids_restore.unsqueeze(-1).expand(-1, -1, D)
    x_full = torch.gather(x_concat, dim=1, index=index)
    return x_full


class SprintLuminaNextDit(LuminaNextDit):
    """
    Lumina Next-DiT with custom SPRINT-aligned encoder, middle,
    and decoder layers.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,
        cross_attention_dim: int = 1024,
        base_sequence_length: int = 256,
        eps: float = 1e-5,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        drop_ratio: float = 0.75,
        residual_type: str = "concat_linear",
        cfg_mask_prob: float = 0.1,
    ):
        # Prevent base constructor block initialization
        nn.Module.__init__(self)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.middle_depth = depth - encoder_depth - decoder_depth
        self.drop_ratio = drop_ratio
        self.residual_type = residual_type
        self.cfg_mask_prob = cfg_mask_prob

        # Patch Embedder
        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        # Timestep Embedder
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=256,
            output_dim=hidden_size,
        )

        # Encoder stage (Dense)
        self.encoder_blocks = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    cross_attention_dim=cross_attention_dim,
                    eps=eps,
                    base_sequence_length=base_sequence_length,
                )
                for _ in range(self.encoder_depth)
            ]
        )

        # Middle stage (Sparse)
        self.middle_blocks = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    cross_attention_dim=cross_attention_dim,
                    eps=eps,
                    base_sequence_length=base_sequence_length,
                )
                for _ in range(self.middle_depth)
            ]
        )

        # Decoder stage (Dense)
        self.decoder_blocks = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    cross_attention_dim=cross_attention_dim,
                    eps=eps,
                    base_sequence_length=base_sequence_length,
                )
                for _ in range(self.decoder_depth)
            ]
        )

        # Output projection
        self.norm_out = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self.mask_token = nn.Parameter(torch.zeros(self.hidden_size))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        if self.residual_type == "concat_linear":
            self.renoise_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
            torch.nn.init.xavier_uniform_(self.renoise_linear.weight)
            nn.init.zeros_(self.renoise_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
        x, (H, W) = self.patchify(x)
        x = self.x_embedder(x)

        temb = self.time_embedding(t, x)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(
                hidden_states=x,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        mask_token_3d = self.mask_token.view(1, 1, -1)
        x_clone = x.clone()
        should_drop = self.training and (self.drop_ratio > 0.0)

        # SPRINT token dropping
        if should_drop:
            x_sparse, ids_keep, ids_restore = random_token_drop(x, self.drop_ratio)

            if image_rotary_emb is not None:
                B_emb = image_rotary_emb.shape[0]
                if B_emb == 1:
                    image_rotary_emb_expanded = image_rotary_emb.expand(
                        x.shape[0], -1, -1
                    )
                else:
                    image_rotary_emb_expanded = image_rotary_emb
                D_emb = image_rotary_emb_expanded.shape[-1]
                image_rotary_emb_sparse = torch.gather(
                    image_rotary_emb_expanded,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).expand(-1, -1, D_emb),
                )
            else:
                image_rotary_emb_sparse = None
        else:
            x_sparse = x
            image_rotary_emb_sparse = image_rotary_emb

        # Middle blocks
        for block in self.middle_blocks:
            x_sparse = block(
                hidden_states=x_sparse,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb_sparse,
            )

        # Sequence restoration
        if should_drop:
            x = restore_full_sequence(x_sparse, ids_restore, mask_token_3d)
        else:
            x = x_sparse

        # SPRINT Path-drop CFG mask (training only)
        if self.training and self.cfg_mask_prob > 0:
            B_sz = x.shape[0]
            sample_mask = torch.rand(B_sz, device=x.device) < self.cfg_mask_prob
            mask_tokens_expanded = mask_token_3d.expand(B_sz, x.shape[1], x.shape[2])
            x = torch.where(
                sample_mask.unsqueeze(1).unsqueeze(2), mask_tokens_expanded, x
            )

        # SPRINT Residual Fusion
        if should_drop and self.residual_type == "concat_linear":
            x = torch.cat([x, x_clone], dim=-1)
            x = self.renoise_linear(x)

        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(
                hidden_states=x,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        x = self.norm_out(x)
        x = self.proj_out(x)
        x = self.unpatchify(x, H, W)
        return x


class SprintDualStreamDiT(DualStreamDiT):
    """
    Dual-Stream DiT with custom SPRINT-aligned encoder, middle,
    and decoder layers.
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
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        drop_ratio: float = 0.75,
        drop_target: str = "image",
        residual_type: str = "concat_linear",
        cfg_mask_prob: float = 0.1,
    ):
        # Prevent base constructor block initialization
        nn.Module.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_checkpointing = use_checkpointing
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.middle_depth = depth - encoder_depth - decoder_depth
        self.drop_ratio = drop_ratio
        self.drop_target = drop_target
        self.residual_type = residual_type
        self.cfg_mask_prob = cfg_mask_prob

        # 1. Image Embedder
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # 2. Time Embedder (Used as a prepended token)
        self.time_embedding = TimeEmbeddings(sinusoidal_dim=256, output_dim=hidden_size)
        self.time_token_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

        # 3. Transformer Text Adapter
        self.text_adapter = TransformerTextAdapter(
            in_channels=text_embed_dim,
            hidden_size=hidden_size,
            num_layers=2,
            num_attention_heads=num_heads,
            ffn_expansion_ratio=mlp_ratio,
            use_checkpointing=self.use_checkpointing,
        )

        # 4. 3D RoPE
        head_dim = hidden_size // num_heads
        axes_dims = _default_rope_axes_dims(head_dim)
        self.rope_embedder = MultimodalRopeEmbedder(axes_dims)

        # Encoder blocks
        self.in_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_checkpointing=self.use_checkpointing,
                )
                for _ in range(self.encoder_depth)
            ]
        )

        # Middle blocks
        # long_skip dual stream has a mid block so layers+1
        self.mid_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_checkpointing=self.use_checkpointing,
                )
                for _ in range(self.middle_depth)
            ]
        )

        # Decoder blocks
        self.out_blocks = nn.ModuleList(
            [
                DualStreamDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    eps=eps,
                    use_skip=True,
                    use_checkpointing=self.use_checkpointing,
                )
                for _ in range(self.decoder_depth)
            ]
        )

        self.norm_final = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

        # torch compile squeezes during bw so remove dims
        self.mask_token_image = nn.Parameter(torch.zeros(self.hidden_size))
        self.mask_token_text = nn.Parameter(torch.zeros(self.hidden_size))
        torch.nn.init.normal_(self.mask_token_image, std=0.02)
        torch.nn.init.normal_(self.mask_token_text, std=0.02)

        if self.residual_type == "concat_linear":
            self.renoise_linear_image = nn.Linear(
                self.hidden_size * 2, self.hidden_size
            )
            self.renoise_linear_text = nn.Linear(self.hidden_size * 2, self.hidden_size)
            torch.nn.init.xavier_uniform_(self.renoise_linear_image.weight)
            torch.nn.init.xavier_uniform_(self.renoise_linear_text.weight)
            nn.init.zeros_(self.renoise_linear_image.bias)
            nn.init.zeros_(self.renoise_linear_text.bias)

        self._zero_initialize_output()

    def _drop_tokens(self, tokens, freqs):
        (
            tokens_sparse,
            ids_keep,
            ids_restore,
        ) = random_token_drop(tokens, self.drop_ratio)

        cos, sin = freqs
        D_half = cos.shape[-1]
        cos_sparse = torch.gather(
            cos,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D_half),
        )
        sin_sparse = torch.gather(
            sin,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D_half),
        )
        freqs_sparse = (cos_sparse, sin_sparse)
        return tokens_sparse, freqs_sparse, ids_keep, ids_restore

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

        image_tokens = self.x_embedder(x).flatten(2).transpose(1, 2)

        t_emb = self.time_embedding(t, x)
        time_token = self.time_token_proj(t_emb).unsqueeze(1)

        text_tokens = self.text_adapter(
            encoder_hidden_states, attention_mask=attention_mask
        )
        text_tokens = torch.cat([time_token, text_tokens], dim=1)

        time_mask = torch.ones((bsz, 1), dtype=torch.bool, device=x.device)
        text_mask = torch.cat([time_mask, attention_mask.bool()], dim=1)

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
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
            )
            skips.append((image_tokens, text_tokens))

        mask_token_image_3d = self.mask_token_image.view(1, 1, -1)
        mask_token_text_3d = self.mask_token_text.view(1, 1, -1)

        image_tokens_clone = image_tokens.clone()
        text_tokens_clone = text_tokens.clone()
        should_drop = self.training and (self.drop_ratio > 0.0)

        # Drop tokens based on targets
        if should_drop:
            if self.drop_target in ["image", "both"]:
                (
                    image_tokens_sparse,
                    image_freqs_sparse,
                    image_ids_keep,
                    image_ids_restore,
                ) = self._drop_tokens(image_tokens, image_freqs)
            else:
                image_tokens_sparse = image_tokens
                image_freqs_sparse = image_freqs

            if self.drop_target in ["text", "both"]:
                (
                    text_tokens_sparse,
                    text_freqs_sparse,
                    text_ids_keep,
                    text_ids_restore,
                ) = self._drop_tokens(text_tokens, text_freqs)
                text_mask_sparse = torch.gather(text_mask, dim=1, index=text_ids_keep)
            else:
                text_tokens_sparse = text_tokens
                text_freqs_sparse = text_freqs
                text_mask_sparse = text_mask
        else:
            image_tokens_sparse = image_tokens
            image_freqs_sparse = image_freqs
            text_tokens_sparse = text_tokens
            text_freqs_sparse = text_freqs
            text_mask_sparse = text_mask

        # Middle blocks
        for block in self.mid_blocks:
            image_tokens_sparse, text_tokens_sparse = block(
                image_tokens_sparse,
                text_tokens_sparse,
                image_freqs_sparse,
                text_freqs_sparse,
                text_mask_sparse,
            )

        # Restore sequences
        if should_drop:
            if self.drop_target in ["image", "both"]:
                image_tokens = restore_full_sequence(
                    image_tokens_sparse,
                    image_ids_restore,
                    mask_token_image_3d,
                )
            else:
                image_tokens = image_tokens_sparse

            if self.drop_target in ["text", "both"]:
                text_tokens = restore_full_sequence(
                    text_tokens_sparse,
                    text_ids_restore,
                    mask_token_text_3d,
                )
            else:
                text_tokens = text_tokens_sparse
        else:
            image_tokens = image_tokens_sparse
            text_tokens = text_tokens_sparse

        # SPRINT Path-drop CFG mask (training only)
        # TODO: implement pdg sampling
        if self.training and self.cfg_mask_prob > 0:
            sample_mask = (
                torch.rand(bsz, device=image_tokens.device) < self.cfg_mask_prob
            )
            if self.drop_target in ["image", "both"]:
                mask_tokens_expanded = mask_token_image_3d.expand(
                    bsz, image_tokens.shape[1], self.hidden_size
                )
                image_tokens = torch.where(
                    sample_mask.unsqueeze(1).unsqueeze(2),
                    mask_tokens_expanded,
                    image_tokens,
                )
            if self.drop_target in ["text", "both"]:
                mask_tokens_expanded = mask_token_text_3d.expand(
                    bsz, text_tokens.shape[1], self.hidden_size
                )
                text_tokens = torch.where(
                    sample_mask.unsqueeze(1).unsqueeze(2),
                    mask_tokens_expanded,
                    text_tokens,
                )

        # Residual Fusion
        if should_drop and self.residual_type == "concat_linear":
            if self.drop_target in ["image", "both"]:
                image_tokens = torch.cat([image_tokens, image_tokens_clone], dim=-1)
                image_tokens = self.renoise_linear_image(image_tokens)
            if self.drop_target in ["text", "both"]:
                text_tokens = torch.cat([text_tokens, text_tokens_clone], dim=-1)
                text_tokens = self.renoise_linear_text(text_tokens)

        # Decoder blocks
        for block in self.out_blocks:
            skip_tensors = skips.pop() if skips else None
            image_tokens, text_tokens = block(
                image_tokens,
                text_tokens,
                image_freqs,
                text_freqs,
                text_mask,
                skip=skip_tensors,
            )

        tokens = self.proj_out(self.norm_final(image_tokens))

        tokens = tokens.reshape(bsz, h_patches, w_patches, p, p, self.out_channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(bsz, self.out_channels, H, W)

        return tokens
