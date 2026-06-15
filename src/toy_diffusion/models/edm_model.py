import torch
import torch.nn as nn
import numpy as np


class EDMPreconditioner(nn.Module):
    """
    Universal EDM Preconditioner.
    Wraps a neural network F_theta to function as a Denoiser D_theta(x; sigma) -> x_0.

    Supports multiple prediction targets by adjusting the skip/out coefficients.
    Regardless of the target, the output of this module is always the estimated clean image x_0.
    """

    def __init__(self, model, prediction_target="edm", sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.prediction_target = prediction_target

        valid_targets = ["x", "eps", "v"]
        if prediction_target not in valid_targets:
            raise ValueError(
                f"Unknown prediction target: {prediction_target}. Supported: {valid_targets}"
            )

    def get_scalings(self, sigma):
        """
        Returns c_skip, c_out, c_in, c_noise based on prediction target.
        """
        # c_in ensures the network input is roughly unit variance
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        if self.prediction_target == "x":
            # Network predicts a mix of x and n, optimized for unit variance output
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

        elif self.prediction_target == "eps":
            # Epsilon Prediction: x_0 = x - sigma * eps
            c_skip = torch.ones_like(sigma)
            c_out = -sigma

        elif self.prediction_target == "v":
            # assumes the network predicts v, and maps it to x_0
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = -sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

        return c_skip, c_out, c_in, c_noise

    def forward(self, x, sigma, **kwargs):
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))

        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)

        x_in = c_in * x

        # c_noise is passed as the timestep embedding
        F_x = self.model(x_in, c_noise.view(-1), **kwargs)

        D_x = c_skip * x + c_out * F_x
        return D_x
