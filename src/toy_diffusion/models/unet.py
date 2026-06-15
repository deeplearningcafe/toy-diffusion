import torch
import torch.nn as nn

from toy_diffusion.models.layers import (
    TimeEmbeddings,
    Transformer2DBlock,
    ResnetBlock2D,
    Upsample2D,
)


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        block_out_channels: list[int] = [320, 640, 1280, 1280],
        transformer_layers_per_block: int | list[int] = 1,
        cross_attention_dim: int = None,
        num_attention_heads: int | list[int] = 8,
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
        # time embed dim is last block dim
        time_embed_dim = self.block_out_channels[-1]
        self.time_embedding = TimeEmbeddings(
            sinusoidal_dim=self.block_out_channels[0],
            output_dim=time_embed_dim,
        )

        # down_blocks is Resnet  ->  CrossAtten -> Downsample except last block is just Resnet
        # is the AttentionBlock the one with the double the channels for the up block in the unet
        # and is the resnet the one how changes the block channels
        self.down_blocks = nn.ModuleList([])
        out_channels = self.block_out_channels[0]
        for i in range(len(self.block_out_channels)):
            in_channels = out_channels
            out_channels = self.block_out_channels[i]
            is_final_block = i == len(self.block_out_channels) - 1

            num_heads = self.num_attention_heads[i]
            num_trans_layers = self.transformer_layers_per_block[i]

            # Create a regular ModuleList for resnets and attentions instead of ModuleDict
            resnets = nn.ModuleList([])
            attentions = nn.ModuleList([])
            for j in range(self.layers_per_block):
                # First resnet in block handles channel changes
                res_input_channels = in_channels if j == 0 else out_channels
                resnets.append(
                    ResnetBlock2D(
                        in_channels=res_input_channels,
                        out_channels=out_channels,
                        time_embeddings_channels=time_embed_dim,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        dropout=self.dropout,
                    )
                )
                # Add attn except final
                if not is_final_block:
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
                        )
                    )
            down_block = nn.ModuleDict(
                {
                    "resnets": resnets,
                    "attentions": attentions,
                }
            )
            if not is_final_block:
                # downsampler is just a simple conv2d
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

        mid_num_heads = self.num_attention_heads[-1]
        mid_num_trans_layers = self.transformer_layers_per_block[-1]

        # always 1 resnets and num_layers=1 so res -> attn -> res
        self.mid_block = nn.ModuleDict(
            {
                "resnets": nn.ModuleList(
                    [
                        ResnetBlock2D(
                            in_channels=self.block_out_channels[-1],
                            out_channels=self.block_out_channels[-1],
                            time_embeddings_channels=time_embed_dim,
                            norm_num_groups=self.norm_num_groups,
                            eps=self.norm_eps,
                            dropout=self.dropout,
                        ),
                        ResnetBlock2D(
                            in_channels=self.block_out_channels[-1],
                            out_channels=self.block_out_channels[-1],
                            time_embeddings_channels=time_embed_dim,
                            norm_num_groups=self.norm_num_groups,
                            eps=self.norm_eps,
                            dropout=self.dropout,
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
                        )
                    ]
                ),
            }
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        reversed_num_attention_heads = list(reversed(self.num_attention_heads))
        reversed_transformer_layers = list(reversed(self.transformer_layers_per_block))

        out_channels = reversed_block_out_channels[0]
        # num layers is +1 -> decoder higher param than encoder
        upsampling_num_layers = self.layers_per_block + 1

        for i in range(len(reversed_block_out_channels)):
            prev_out_channels = out_channels
            out_channels = reversed_block_out_channels[i]
            in_channels = reversed_block_out_channels[
                min(i + 1, len(reversed_block_out_channels) - 1)
            ]
            is_first_block = i == 0

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
                    ResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,
                        out_channels=out_channels,
                        time_embeddings_channels=time_embed_dim,
                        norm_num_groups=self.norm_num_groups,
                        eps=self.norm_eps,
                        dropout=self.dropout,
                    )
                )

                # Add attention blocks (in up blocks they come after resnets)
                if not is_first_block:
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
                        )
                    )
            up_block = nn.ModuleDict({"resnets": resnets, "attentions": attentions})

            # No upsampler needed for the last block
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

        # out conv and norm
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
        """
        Crucial for eps-prediction: Initialize the final output layer to zero.
        This ensures the model initially predicts 0 noise, which is the mean
        of the target N(0, I) distribution, preventing huge initial loss spikes.
        """
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

        for block in self.down_blocks:
            for resnet in block.resnets:
                nn.init.zeros_(resnet.conv2.weight)
                if resnet.conv2.bias is not None:
                    nn.init.zeros_(resnet.conv2.bias)

        for resnet in self.mid_block.resnets:
            nn.init.zeros_(resnet.conv2.weight)
            if resnet.conv2.bias is not None:
                nn.init.zeros_(resnet.conv2.bias)

        for block in self.up_blocks:
            for resnet in block.resnets:
                nn.init.zeros_(resnet.conv2.weight)
                if resnet.conv2.bias is not None:
                    nn.init.zeros_(resnet.conv2.bias)

    def forward(self, x, t, encoder_hidden_states=None, attention_mask=None):
        t_emb = self.time_embedding(t, x)
        x = self.conv_in(x)

        down_block_res_x = (x,)

        for i, down_block in enumerate(self.down_blocks):
            output_states = ()
            if i != (len(self.block_out_channels) - 1):
                for resnet, attention in zip(down_block.resnets, down_block.attentions):
                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )

                    output_states = output_states + (x,)

                for downsampler in down_block.downsamplers:
                    x = self._checkpoint(downsampler, x)
                output_states = output_states + (x,)

            # last block has no attn nor downsample
            else:
                for resnet in down_block.resnets:
                    x = self._checkpoint(resnet, x, t_emb)
                    output_states += (x,)

            down_block_res_x += output_states

        # mid_block has 1 resnet more than attn
        x = self._checkpoint(self.mid_block.resnets[0], x, t_emb)
        for attention, resnet in zip(
            self.mid_block.attentions, self.mid_block.resnets[1:]
        ):
            x = attention(x, encoder_hidden_states, attention_mask=attention_mask)
            x = self._checkpoint(resnet, x, t_emb)

        for i, up_block in enumerate(self.up_blocks):
            # each resnet gets a res input
            res_x_tuple = down_block_res_x[-len(up_block.resnets) :]
            down_block_res_x = down_block_res_x[: -len(up_block.resnets)]

            # first block no attn
            if i == 0:
                for resnet in up_block.resnets:
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    # concat on channels
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)
                for upsampler in up_block.upsamplers:
                    x = self._checkpoint(upsampler, x)

            # except last block
            elif i != len(self.up_blocks) - 1:
                for resnet, attention in zip(up_block.resnets, up_block.attentions):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    # concat on channels
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )
                for upsampler in up_block.upsamplers:
                    x = self._checkpoint(upsampler, x)

            # last block has no upsampler
            else:
                for resnet, attention in zip(up_block.resnets, up_block.attentions):
                    res_x = res_x_tuple[-1]
                    res_x_tuple = res_x_tuple[:-1]
                    # concat on channels
                    x = torch.cat([x, res_x], dim=1)

                    x = self._checkpoint(resnet, x, t_emb)
                    x = attention(
                        x, encoder_hidden_states, attention_mask=attention_mask
                    )

        x = self._checkpoint(self.conv_norm_out, x)
        x = self._checkpoint(self.conv_act, x)
        x = self.conv_out(x)

        return x

    def _checkpoint(self, module, *args):
        """Helper fn to apply activation checkpointing"""
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=True)
        else:
            return module(*args)
