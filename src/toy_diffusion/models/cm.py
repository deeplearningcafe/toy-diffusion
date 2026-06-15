import torch
import torch.nn as nn


class ConsistencyPreconditioner(nn.Module):
    """
    Consistency Model Preconditioner.
    Wraps a neural network F_theta to function as a Consistency Model f_theta(x, t).
    Enforces the boundary condition f_theta(x, epsilon) = x.
    """

    def __init__(self, model, sigma_data=0.5, epsilon=0.002):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def get_scalings(self, sigma):
        """
        Returns c_skip, c_out, c_in, c_noise based on the Consistency Models paper.
        """
        # Eq. in Appendix C of Consistency Models paper
        c_skip = self.sigma_data**2 / ((sigma - self.epsilon) ** 2 + self.sigma_data**2)
        c_out = (
            self.sigma_data
            * (sigma - self.epsilon)
            / (self.sigma_data**2 + sigma**2).sqrt()
        )
        c_in = 1.0 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = 0.25 * sigma.log()

        return c_skip, c_out, c_in, c_noise

    def forward(self, x, sigma, **kwargs):
        sigma = sigma.view(-1, *([1] * (x.ndim - 1)))

        c_skip, c_out, c_in, c_noise = self.get_scalings(sigma)

        # Precondition Input
        x_in = c_in * x

        F_x = self.model(x_in, c_noise.view(-1), **kwargs)

        # Precondition Output to enforce boundary condition
        D_x = c_skip * x + c_out * F_x
        return D_x
