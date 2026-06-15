import torch
import torch.nn as nn

from toy_diffusion.models.layers import (
    Attention,
    Feedforward,
    TimeEmbeddings,
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
        self.norm = nn.RMSNorm(embedding_dim, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)

        x = self.norm(x) * (1 + scale_msa[:, None, :])

        return x, gate_msa, scale_mlp, gate_mlp


class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock implementing Sandwich Normalization.
    Adds RMSNorm both before and after each attention and MLP layer.
    Pre-norm and post-norm are placed before the scale operation.
    A tanh gating is applied to the residual branch to prevent
    uncontrollable growth of network activations.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        eps: float = 1e-5,
        base_sequence_length: int = 256,
    ):
        super().__init__()

        # Sandwich Norm: Pre-norms
        self.norm1 = LuminaRMSNormZero(embedding_dim=dim, eps=eps)
        self.norm1_context = (
            nn.RMSNorm(cross_attention_dim, eps=eps) if cross_attention_dim else None
        )

        # Self Attention
        self.attn1 = Attention(
            in_channels=dim,
            num_attention_heads=num_attention_heads,
            kv_num_heads=num_kv_heads,
            qk_norm="rms_norm",
            eps=eps,
            base_sequence_length=base_sequence_length,
        )

        # Sandwich Norm: Post-norms for Attention
        self.norm2 = nn.RMSNorm(dim, eps=eps)

        # Feedforward and its Sandwich Norms
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.feed_forward = Feedforward(in_channels=dim)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
        residual = hidden_states

        # 1. Self-attention with Sandwich Norm
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
            hidden_states, temb
        )

        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)

        # Post-norm + Tanh gating on the residual
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(
            attn_output
        )

        # 3. Feedforward with Sandwich Norm
        mlp_input = self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1))
        mlp_output = self.feed_forward(mlp_input)

        # Post-norm + Tanh gating on the residual
        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
            mlp_output
        )

        return hidden_states


class LuminaNextDit(nn.Module):
    """
    Lumina Next-DiT Architecture.
    Implements a single-stream diffusion transformer using 3D RoPE,
    Sandwich Normalization, and Grouped-Query Attention.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_attention_heads: int = 16,
        num_kv_heads: int = 4,  # GQA: 16 query heads, 4 KV heads
        cross_attention_dim: int = 1024,
        base_sequence_length: int = 256,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Patch Embedder
        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels, out_features=hidden_size
        )

        # Timestep Embedder
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=256,
            output_dim=hidden_size,
        )

        # DiT Blocks
        self.blocks = nn.ModuleList(
            [
                LuminaNextDiTBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    cross_attention_dim=cross_attention_dim,
                    eps=eps,
                    base_sequence_length=base_sequence_length,
                )
                for _ in range(depth)
            ]
        )

        # Output Norm and Projection
        self.norm_out = nn.RMSNorm(hidden_size, eps=eps)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
        return x, (H, W)

    def unpatchify(self, x, H, W):
        B, _, _ = x.shape
        p = self.patch_size
        x = x.view(B, H // p, W // p, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
    ):
        # 1. Patchify and Embed
        x, (H, W) = self.patchify(x)
        x = self.x_embedder(x)

        # 2. Time Embedding
        temb = self.time_embedding(t, x)

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(
                hidden_states=x,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        # 4. Output Projection
        x = self.norm_out(x)
        x = self.proj_out(x)

        # 5. Unpatchify
        x = self.unpatchify(x, H, W)

        return x
