import torch
import torch.nn as nn

from toy_diffusion.models.layers import (
    TimeEmbeddings,
    Transformer2DBlock,
    Upsample2D,
    EfficientResnetBlock2D,
)


class EfficientUnet(nn.Module):
    """
    Implementation of the SnapGen: Taming High-Resolution Text-to-Image Models
    for Mobile Devices with Efficient Architectures and Training paper
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        block_out_channels: list[int] = [256, 512, 896],
        transformer_layers_per_block: int | list[int] = [1, 2, 4],
        cross_attention_dim: int = 256,
        num_attention_heads: int | list[int] = [4, 8, 14],
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-05,
        use_checkpointing: bool = True,
        dropout: float = 0.0,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.cross_attention_dim = cross_attention_dim
        self.layers_per_block = layers_per_block
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.use_checkpointing = use_checkpointing
        self.dropout = dropout
        self.device = device

        if isinstance(num_attention_heads, int):
            self.num_attention_heads = [num_attention_heads] * len(block_out_channels)
        else:
            self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            self.transformer_layers_per_block = [transformer_layers_per_block] * len(
                block_out_channels
            )
        else:
            self.transformer_layers_per_block = transformer_layers_per_block

        # 1. input
        self.conv_in = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # 2. time embeds
        time_embed_dim = self.block_out_channels[-1]
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=self.block_out_channels[0],
            output_dim=time_embed_dim,
        )

        # 3. Down Blocks
        self.down_blocks = nn.ModuleList([])
        out_channels = self.block_out_channels[0]
        for i in range(len(self.block_out_channels)):
            in_channels = out_channels
            out_channels = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1

            # SA is ONLY active in the lowest-resolution stage
            is_lowest_res = is_final_block

            num_heads = self.num_attention_heads[i]
            num_trans_layers = self.transformer_layers_per_block[i]

            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])
            for j in range(self.layers_per_block):
                res_input_channels = in_channels if j == 0 else out_channels
                resnets.append(
                    EfficientResnetBlock2D(
                        in_channels=res_input_channels,
                        out_channels=out_channels,
                        time_embeddings_channels=time_embed_dim,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        dropout=self.dropout,
                        expansion_ratio=2.0,
                    )
                )

                # Insert conditions from 1st stage. ALL blocks get Transformers.
                attentions.append(
                    Transformer2DBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        cross_attention_dim=self.cross_attention_dim,
                        num_attention_heads=num_heads,
                        num_layers=num_trans_layers,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        use_checkpointing=self.use_checkpointing,
                        disable_self_attention=not is_lowest_res,
                        use_rope=True,
                        qk_norm="rms_norm",
                        kv_num_heads=1,
                        ffn_expansion_ratio=3,
                    )
                )

            down_block = nn.ModuleDict(
                {
                    "resnets": resnets,
                    "attentions": attentions,
                }
            )
            if not is_final_block:
                down_block["downsamplers"] = nn.ModuleList(
                    [
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        )
                    ]
                )
            self.down_blocks.append(down_block)

        # 4. Mid Block
        mid_num_heads = self.num_attention_heads[-1]
        mid_num_trans_layers = self.transformer_layers_per_block[-1]

        self.mid_block = nn.ModuleDict(
            {
                "resnets": nn.ModuleList(
                    [
                        EfficientResnetBlock2D(
                            in_channels=self.block_out_channels[-1],
                            out_channels=self.block_out_channels[-1],
                            time_embeddings_channels=time_embed_dim,
                            norm_num_groups=self.norm_num_groups,
                            eps=self.norm_eps,
                            dropout=self.dropout,
                            expansion_ratio=2.0,
                        ),
                        EfficientResnetBlock2D(
                            in_channels=self.block_out_channels[-1],
                            out_channels=self.block_out_channels[-1],
                            time_embeddings_channels=time_embed_dim,
                            norm_num_groups=self.norm_num_groups,
                            eps=self.norm_eps,
                            dropout=self.dropout,
                            expansion_ratio=2.0,
                        ),
                    ]
                ),
                "attentions": nn.ModuleList(
                    [
                        Transformer2DBlock(
                            in_channels=self.block_out_channels[-1],
                            out_channels=self.block_out_channels[-1],
                            cross_attention_dim=self.cross_attention_dim,
                            num_attention_heads=mid_num_heads,
                            num_layers=mid_num_trans_layers,
                            norm_num_groups=self.norm_num_groups,
                            eps=self.norm_eps,
                            use_checkpointing=self.use_checkpointing,
                            disable_self_attention=False,
                            use_rope=True,
                            qk_norm="rms_norm",
                            kv_num_heads=1,
                            ffn_expansion_ratio=3,
                        )
                    ]
                ),
            }
        )

        # 5. Up Blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        reversed_num_attention_heads = list(reversed(self.num_attention_heads))
        reversed_transformer_layers = list(reversed(self.transformer_layers_per_block))

        out_channels = reversed_block_out_channels[0]
        upsampling_num_layers = self.layers_per_block + 1

        for i in range(len(reversed_block_out_channels)):
            prev_out_channels = out_channels
            out_channels = reversed_block_out_channels[i]
            in_channels = reversed_block_out_channels[
                min(i + 1, len(reversed_block_out_channels) - 1)
            ]

            is_lowest_res = i == 0

            num_heads = reversed_num_attention_heads[i]
            num_trans_layers = reversed_transformer_layers[i]

            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])
            for j in range(upsampling_num_layers):
                res_skip_channels = (
                    in_channels if (j == upsampling_num_layers - 1) else out_channels
                )
                resnet_in_channels = prev_out_channels if j == 0 else out_channels
                resnets.append(
                    EfficientResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,
                        out_channels=out_channels,
                        time_embeddings_channels=time_embed_dim,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        dropout=self.dropout,
                        expansion_ratio=2.0,
                    )
                )

                attentions.append(
                    Transformer2DBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        cross_attention_dim=self.cross_attention_dim,
                        num_attention_heads=num_heads,
                        num_layers=num_trans_layers,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        use_checkpointing=self.use_checkpointing,
                        disable_self_attention=not is_lowest_res,
                        use_rope=True,
                        qk_norm="rms_norm",
                        kv_num_heads=1,
                        ffn_expansion_ratio=3,
                    )
                )

            up_block = nn.ModuleDict({"resnets": resnets, "attentions": attentions})

            if i < len(self.block_out_channels) - 1:
                up_block["upsamplers"] = nn.ModuleList(
                    [
                        Upsample2D(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            use_conv=False,
                        )
                    ]
                )

            self.up_blocks.append(up_block)

        # 6. Out Conv
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=self.norm_num_groups,
            eps=self.norm_eps,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=self.block_out_channels[0],
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

        self._zero_initialize_output()

    def _zero_initialize_output(self):
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        for block in self.down_blocks:
            for resnet in block.resnets:
                nn.init.zeros_(resnet.pw2.weight)
                if resnet.pw2.bias is not None:
                    nn.init.zeros_(resnet.pw2.bias)

        for resnet in self.mid_block.resnets:
            nn.init.zeros_(resnet.pw2.weight)
            if resnet.pw2.bias is not None:
                nn.init.zeros_(resnet.pw2.bias)

        for block in self.up_blocks:
            for resnet in block.resnets:
                nn.init.zeros_(resnet.pw2.weight)
                if resnet.pw2.bias is not None:
                    nn.init.zeros_(resnet.pw2.bias)

    def forward(self, x, t, encoder_hidden_states=None, attention_mask=None):
        t_emb = self.time_embedding(t, x)
        x = self.conv_in(x)

        down_block_res_x = (x,)

        for i, down_block in enumerate(self.down_blocks):
            output_states = ()
            for resnet, attention in zip(down_block.resnets, down_block.attentions):
                x = self._checkpoint(resnet, x, t_emb)
                x = attention(x, encoder_hidden_states, attention_mask=attention_mask)
                output_states = output_states + (x,)

            if "downsamplers" in down_block:
                for downsampler in down_block.downsamplers:
                    x = self._checkpoint(downsampler, x)
                output_states = output_states + (x,)

            down_block_res_x += output_states

        x = self._checkpoint(self.mid_block.resnets[0], x, t_emb)
        for attention, resnet in zip(
            self.mid_block.attentions, self.mid_block.resnets[1:]
        ):
            x = attention(x, encoder_hidden_states, attention_mask=attention_mask)
            x = self._checkpoint(resnet, x, t_emb)

        for i, up_block in enumerate(self.up_blocks):
            res_x_tuple = down_block_res_x[-len(up_block.resnets) :]
            down_block_res_x = down_block_res_x[: -len(up_block.resnets)]

            for resnet, attention in zip(up_block.resnets, up_block.attentions):
                res_x = res_x_tuple[-1]
                res_x_tuple = res_x_tuple[:-1]
                x = torch.cat([x, res_x], dim=1)

                x = self._checkpoint(resnet, x, t_emb)
                x = attention(x, encoder_hidden_states, attention_mask=attention_mask)

            if "upsamplers" in up_block:
                for upsampler in up_block.upsamplers:
                    x = self._checkpoint(upsampler, x)

        x = self._checkpoint(self.conv_norm_out, x)
        x = self._checkpoint(self.conv_act, x)
        x = self.conv_out(x)

        return x

    def _checkpoint(self, module, *args):
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=True)
        else:
            return module(*args)

