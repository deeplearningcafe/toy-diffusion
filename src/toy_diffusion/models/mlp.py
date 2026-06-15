import torch
import torch.nn as nn
import math
from toy_diffusion.models.layers import SinusoidalPosEmb


class FlowMLP(nn.Module):
    """
    Simple MLP for Flow Matching.
    Input: x (D) + t (1)
    Output: prediction (D) - can be x, eps, or v depending on config
    """

    def __init__(
        self,
        data_dim,
        hidden_dim=256,
        num_layers=5,
        norm_num_groups=32,
        activation=nn.ReLU,
    ):
        super().__init__()
        # Input dim is D + 1 (for time)
        layers = [nn.Linear(data_dim + 1, hidden_dim), activation()]

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(
                nn.GroupNorm(num_groups=norm_num_groups, num_channels=hidden_dim)
            )
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_embed = t.view(-1, 1)
        # Concatenate x and t
        x_in = torch.cat([x, t_embed], dim=1)
        return self.net(x_in)


class ResModel(nn.Module):
    """
    A simple MLP that mimics the behavior of a time-conditioned network.
    Includes residual connections to allow learning identity easily (crucial for x-pred).
    """

    def __init__(
        self,
        data_dim=2,
        hidden_dim=256,
        time_embed_dim=64,
        num_layers=5,
        norm_num_groups=32,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # follow resnet structure:
        # norm->act->conv->norm->act->conv
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(num_groups=norm_num_groups, num_channels=hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GroupNorm(num_groups=norm_num_groups, num_channels=hidden_dim),
                    activation(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=hidden_dim
        )

        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = activation()

    def forward(self, x, t):
        # t is expected to be [0, 1]
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)

        h = x_emb + t_emb

        for block in self.blocks:
            h = h + block(h)

        h = self.output_norm(h)
        return self.output_proj(self.act(h))


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.abs().max() <= 10.0:
            t = t * 1000.0

        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        return alpha * x


class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: list[int], num_timesteps: int
    ):
        super().__init__()
        """
        From Kaist diffusion course.
        Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        self.tlins = nn.ModuleList()
        dims = [dim_in] + dim_hids + [dim_out]
        for i in range(len(dims) - 1):
            tlin = TimeLinear(dims[i], dims[i + 1], num_timesteps)
            self.tlins.append(tlin)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        for i in range(len(self.tlins)):
            x = self.tlins[i](x, t)
            if i != len(self.tlins) - 1:
                x = torch.nn.functional.relu(x)
        return x


class DDGANGenerator(nn.Module):
    """
    Generator for DD-GAN. Predicts x_0 given x_t, t, and a latent variable z.
    The latent z enables modeling multimodal denoising distributions.
    """

    def __init__(self, data_dim=2, latent_dim=4, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_mlp = TimeEmbedding(time_embed_dim)

        self.fc1 = nn.Linear(data_dim + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, data_dim)

        self.t1 = nn.Linear(time_embed_dim, hidden_dim)
        self.t2 = nn.Linear(time_embed_dim, hidden_dim)
        self.t3 = nn.Linear(time_embed_dim, hidden_dim)

    def forward(self, x, t, z=None):
        if z is None:
            z = torch.randn(x.shape[0], self.latent_dim, device=x.device)
        t_emb = self.time_mlp(t)

        h = torch.cat([x, z], dim=1)
        h = torch.nn.functional.silu(self.fc1(h) + self.t1(t_emb))
        h = torch.nn.functional.silu(self.fc2(h) + self.t2(t_emb))
        h = torch.nn.functional.silu(self.fc3(h) + self.t3(t_emb))
        return self.fc4(h)


class DDGANDiscriminator(nn.Module):
    """
    Discriminator for DD-GAN. Distinguishes between True and Fake
    denoising steps conditioned on the noisier state x_curr and time t.
    """

    def __init__(self, data_dim=2, hidden_dim=256, time_embed_dim=64):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_embed_dim)

        self.fc1 = nn.Linear(data_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

        self.t1 = nn.Linear(time_embed_dim, hidden_dim)
        self.t2 = nn.Linear(time_embed_dim, hidden_dim)
        self.t3 = nn.Linear(time_embed_dim, hidden_dim)

    def forward(self, x_next, x_curr, t):
        t_emb = self.time_mlp(t)

        h = torch.cat([x_next, x_curr], dim=1)
        h = torch.nn.functional.leaky_relu(self.fc1(h) + self.t1(t_emb), 0.2)
        h = torch.nn.functional.leaky_relu(self.fc2(h) + self.t2(t_emb), 0.2)
        h = torch.nn.functional.leaky_relu(self.fc3(h) + self.t3(t_emb), 0.2)
        return self.fc4(h)
