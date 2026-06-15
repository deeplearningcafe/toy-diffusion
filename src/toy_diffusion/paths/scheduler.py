import torch
from abc import ABC, abstractmethod


class Schedule(ABC):
    """
    Abstract base class for the diffusion schedule.
    Returns alpha, sigma, and their time derivatives.
    Assumes t goes from 0 (Noise) to 1 (Data).
    """

    def __init__(self, device="cpu"):
        self.device = device

    @abstractmethod
    def get_coefficients(self, t: torch.Tensor):
        """
        Returns:
            alpha, sigma, d_alpha, d_sigma
        """
        pass

    @abstractmethod
    def get_scheduler_type(self):
        """
        Returns:
            name of the scheduler
        """
        pass


class LinearSchedule(Schedule):
    """
    Standard Rectified Flow / Flow Matching Schedule.
    t=0 (Noise) -> t=1 (Data).

    alpha(t) = t
    sigma(t) = 1 - t
    """

    def get_coefficients(self, t: torch.Tensor):
        # alpha = t, sigma = 1-t
        alpha = t
        sigma = 1.0 - t

        # d_alpha/dt = 1
        d_alpha = torch.ones_like(t)
        # d_sigma/dt = -1
        d_sigma = -torch.ones_like(t)

        return alpha, sigma, d_alpha, d_sigma

    def get_scheduler_type(self):
        return "linear"


class DDPMSchedule(Schedule):
    """
    Adapts a discrete DDPM beta schedule to a continuous Flow Matching schedule.

    Mapping:
        FM t=0 -> DDPM Step T (Pure Noise)
        FM t=1 -> DDPM Step 0 (Clean Data)

    We use numerical differentiation (finite differences) to compute d_alpha/dt.
    """

    def __init__(
        self, device="cpu", num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012
    ):
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
            )
            ** 2
        ).to(device)

        # Alphas (Step signal scale squared)
        # betas are already the variance
        # This is the variance of the signal retention per step
        self.alphas = 1.0 - self.betas

        # Alphas Cumprod (Cumulative signal variance) -> \bar{alpha}^2
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # alpha(t) = sqrt(alpha_bar)
        self.alpha_arr = torch.sqrt(self.alphas_cumprod).flip(0)

        # sigma(t) = sqrt(1 - alpha_bar)
        self.sigma_arr = torch.sqrt(1.0 - self.alphas_cumprod).flip(0)

        # dt = 1 / T
        dt = 1.0 / (self.num_train_timesteps - 1)

        self.d_alpha_arr = torch.gradient(self.alpha_arr, spacing=dt)[0]
        self.d_sigma_arr = torch.gradient(self.sigma_arr, spacing=dt)[0]

    def _interpolate(self, t, arr):
        # TODO: use the formula from 4.3.2: b(t) = b_min + t(b_max - b_min), t in [0,1]
        # Map t [0, 1] to indices [0, T-1]
        # Clamp t to ensure indices are valid
        t = t.clamp(0.0, 1.0)
        float_idx = t * (self.num_train_timesteps - 1)
        idx_floor = (
            float_idx.long().clamp(0, self.num_train_timesteps - 2).to(self.device)
        )
        idx_ceil = (
            (idx_floor + 1).clamp(0, self.num_train_timesteps - 1).to(self.device)
        )

        w = float_idx.to(self.device) - idx_floor.float()

        val_floor = arr[idx_floor]
        val_ceil = arr[idx_ceil]
        return val_floor * (1 - w) + val_ceil * w

    def get_coefficients(self, t: torch.Tensor):
        alpha = self._interpolate(t, self.alpha_arr)
        sigma = self._interpolate(t, self.sigma_arr)
        d_alpha = self._interpolate(t, self.d_alpha_arr)
        d_sigma = self._interpolate(t, self.d_sigma_arr)
        return alpha, sigma, d_alpha, d_sigma

    def get_scheduler_type(self):
        return "ddpm"


class VESchedule(Schedule):
    """
    Karras EDM Variance Exploding Schedule.

    Mapping:
        t=0 -> sigma_max (Noise)
        t=1 -> sigma_min (Data)

    Uses the polynomial schedule for the mapping sigma(t).
    """

    def __init__(self, device="cpu", sigma_min=0.002, sigma_max=80.0, rho=7.0):
        super().__init__(device)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.inv_rho = 1.0 / rho
        self.sigma_min_inv_rho = sigma_min**self.inv_rho
        self.sigma_max_inv_rho = sigma_max**self.inv_rho

    def get_coefficients(self, t: torch.Tensor):
        """
        Returns alpha, sigma, d_alpha, d_sigma for the VE path.
        In VE: x_t = x_0 + sigma(t) * eps
        Therefore: alpha(t) = 1.0
        """
        # Formula: sigma(t) = (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        t = t.clamp(0.0, 1.0)

        # Linear interpolation in rho-space
        # t=0 -> sigma_max, t=1 -> sigma_min
        val = self.sigma_max_inv_rho + t * (
            self.sigma_min_inv_rho - self.sigma_max_inv_rho
        )
        sigma = val**self.rho

        # Alpha is constant 1.0 for VE
        alpha = torch.ones_like(t)

        # d_alpha / dt = 0
        d_alpha = torch.zeros_like(t)

        # d_sigma / dt via chain rule
        # d/dt [ (A + t(B-A))^rho ] = rho * (A + t(B-A))^(rho-1) * (B-A)
        diff = self.sigma_min_inv_rho - self.sigma_max_inv_rho
        d_sigma = self.rho * (val ** (self.rho - 1)) * diff

        return alpha, sigma, d_alpha, d_sigma

    def get_scheduler_type(self):
        return "karras_edm"
