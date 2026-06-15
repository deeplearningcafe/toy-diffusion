import torch
import torch.nn as nn
import math
from toy_diffusion.models.edm_model import EDMPreconditioner
from toy_diffusion.paths.scheduler import Schedule


def logit_normal_sample(n, device, mean=0.0, std=1.0, shift=1.0):
    """
    Samples t from a Logit-Normal distribution.
    Applies timeshift mathematically by adding log(shift) to the mean.
    """
    # For t=0 (Noise), FLUX shifts logit by log(s). Because our t is inverted,
    # logit(1-t) = -logit(t), so we shift the mean by -log(s).
    mean = -math.log(shift) if shift != 1.0 else 0.0
    s = torch.randn(n, device=device) * std + mean
    return torch.sigmoid(s)


def log_normal_sigma(n, device, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    rnd_normal = torch.randn(n, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    return sigma


def uniform_timesteps(n, device):
    """Standard Uniform sampling t ~ U[0, 1]"""
    timesteps = torch.rand((n,), device=device)
    return timesteps


def get_timestep_sampling_fn(timestep_sampling):
    if timestep_sampling == "logit-normal":

        def sample_fn(n, device, shift=1.0):
            return logit_normal_sample(n, device, mean=0.0, std=1.0, shift=shift)

        return sample_fn
    elif timestep_sampling == "uniform":

        def sample_fn(n, device, shift=1.0):
            return uniform_timesteps(n, device)

        return sample_fn


def min_snr_gamma(t, snr, gamma=5.0):
    """
    It gives weight of 1 to the noise steps: snr < gamma
    and gamma/snr < 1 weight to cleaner steps: snr > gamma
    """
    min_snr = torch.minimum(snr, torch.full_like(snr, gamma))

    # min(SNR, gamma) / SNR
    snr_safe = snr.clamp(min=1e-5)
    snr_weight = torch.div(min_snr, snr_safe).float().to(t.device)
    return snr_weight


def debias_snr(t, snr):
    snr_weight = 1.0 / torch.sqrt(snr)
    return snr_weight.to(t.device)


# logic from https://github.com/bluvoll/sd-scripts-f2vae/blob/main/library/train_util.py
def euclidean_optimal_transport(
    X: torch.Tensor, Y: torch.Tensor, backend: str = "auto"
):
    """Compute an optimal assignment under Euclidean (L2) distance.
    Cosine assumes data perfectly normalized, for latents works buts
    for raw images better L2
    """
    # X and Y are shape (B, D)
    # torch.cdist computes the pairwise L2 distance matrix of shape (B, B)
    cost = torch.cdist(X, Y, p=2.0)

    if backend == "cuda":
        return _cuda_assignment(cost)
    if backend == "scipy":
        return _scipy_assignment(cost)

    try:
        return _cuda_assignment(cost)
    except (ImportError, RuntimeError) as exc:
        return _scipy_assignment(cost)


def _cuda_assignment(cost: torch.Tensor):
    from torch_linear_assignment import assignment_to_indices, batch_linear_assignment  # type: ignore

    assignment = batch_linear_assignment(cost.unsqueeze(0))
    row_idx, col_idx = assignment_to_indices(assignment)
    # Squeeze the batch dimension added by unsqueeze(0)
    return cost, (row_idx.squeeze(0), col_idx.squeeze(0))


def _scipy_assignment(cost: torch.Tensor):
    from scipy.optimize import linear_sum_assignment  # type: ignore

    cost_np = cost.to(torch.float32).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    row = torch.from_numpy(row_ind).to(cost.device, torch.long)
    col = torch.from_numpy(col_ind).to(cost.device, torch.long)
    return cost, (row, col)


class GeneralDiffusionLoss(nn.Module):
    """
    Computes diffusion loss allowing for decoupled prediction target and loss target.
    Uses GaussianConditionalPath for ground truth physics (sampling and vector field).
    """

    def __init__(
        self,
        schedule: Schedule,
        prediction_target="v",
        loss_target="v",
        timestep_sampling: str = "logit-normal",
        weight_fn_name: str = None,
        input_perturbation: float = 0.0,
        use_ot: bool = False,
        train_shift: float = 1.0,
        is_conditional: bool = False,
    ):
        super().__init__()
        self.schedule = schedule
        self.prediction_target = prediction_target
        self.loss_target = loss_target
        self.timestep_sampling_fn = get_timestep_sampling_fn(timestep_sampling)
        self.weight_fn_name = weight_fn_name
        self.input_perturbation = input_perturbation
        self.use_ot = use_ot
        self.train_shift = train_shift

        self.set_conditional(is_conditional)

    def set_conditional(self, conditional):
        self.conditional = conditional
        self.model_forward = self._forward_cond if conditional else self._forward_uncond

    def _forward_cond(self, model, z_t, t, prompt):
        text_cond, attention_mask = model["text_enc"](prompt)
        return model["unet"](
            z_t, t, encoder_hidden_states=text_cond, attention_mask=attention_mask
        )

    def _forward_uncond(self, model, z_t, t, prompt):
        if isinstance(model, (dict, nn.ModuleDict)):
            return model["unet"](z_t, t)
        return model(z_t, t)

    def _solve_linear_system(self, pred, z_t, alpha, sigma, d_alpha, d_sigma):
        """
        Solves the system to convert the model prediction to all other forms (x, eps, v).

        System:
        1) z_t = alpha * x + sigma * eps
        2) v   = d_alpha * x + d_sigma * eps
        Based on the JIT paper:
        * v-pred:
            x_0 = sigma * v_pred + z_t
            eps = z_t - alpha * v_pred
        * eps-pred:
            x = (z_t - sigma*eps) / alpha
            v_pred = (z_t - eps_pred)/alpha
        * x-pred:
            eps = (z_t - alpha*x_pred) / sigma
            v_pred = (x_pred - z_t)/sigma


        We have 'pred' which corresponds to self.prediction_target.
        """

        alpha = alpha.clamp(min=1e-5)
        sigma = sigma.clamp(min=1e-5)

        # Calculate Determinant (Wronskian) for v conversion
        # det = alpha * d_sigma - sigma * d_alpha
        det = alpha * d_sigma - sigma * d_alpha
        det_safe = torch.where(det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det)

        if self.prediction_target == "x":
            x_pred = pred
            # eps = (z - alpha * x) / sigma
            eps_pred = (z_t - alpha * x_pred) / sigma
            # v = d_alpha * x + d_sigma * eps
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
            # v_pred = (x_pred - z_t) / sigma

        elif self.prediction_target == "eps":
            eps_pred = pred
            # x = (z - sigma * eps) / alpha
            x_pred = (z_t - sigma * eps_pred) / alpha
            # v = d_alpha * x + d_sigma * eps
            v_pred = d_alpha * x_pred + d_sigma * eps_pred
            # v_pred = (z_t - eps_pred) / alpha

        elif self.prediction_target == "v":
            v_pred = pred
            # Cramer's Rule / Inversion to find x and eps from (z, v)
            # x = (d_sigma * z - sigma * v) / det
            x_pred = (d_sigma * z_t - sigma * v_pred) / det_safe
            # x_pred = sigma * v_pred + z_t

            # eps = (alpha * v - d_alpha * z) / det
            eps_pred = (alpha * v_pred - d_alpha * z_t) / det_safe
            # eps_pred = z_t - alpha * v_pred

        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")

        return x_pred, eps_pred, v_pred

    def forward(self, model, x, prompt=None):
        # Handle Reflow tuples (Noise, Data)
        if isinstance(x, (list, tuple)):
            eps, data = x
            B = data.shape[0]
            device = data.device
        else:
            data = x
            B = data.shape[0]
            device = data.device
            eps = torch.randn_like(data)

            if self.use_ot:
                data_flat = data.view(B, -1)
                eps_flat = eps.view(B, -1)

                _, (row_idx, col_idx) = euclidean_optimal_transport(data_flat, eps_flat)

                # Reorder eps based on the optimal assignment
                eps_sorted = torch.empty_like(eps)
                eps_sorted[row_idx] = eps[col_idx]
                eps = eps_sorted

        t = self.timestep_sampling_fn(B, device, shift=self.train_shift)
        t_view = t.view(-1, *([1] * (data.ndim - 1)))

        alpha, sigma, d_alpha, d_sigma = self.schedule.get_coefficients(t_view)

        noise_perturb = torch.zeros_like(data)
        if self.input_perturbation > 0.0:
            noise_perturb = torch.randn_like(data) * self.input_perturbation
            eps += noise_perturb

        z_t = alpha * data + sigma * eps

        raw_pred = self.model_forward(model, z_t, t, prompt)

        x_pred, eps_pred, v_pred = self._solve_linear_system(
            raw_pred, z_t, alpha, sigma, d_alpha, d_sigma
        )

        if self.loss_target == "x":
            pred = x_pred
            target = data
        elif self.loss_target == "eps":
            pred = eps_pred
            target = eps
        elif self.loss_target == "v":
            pred = v_pred
            target = d_alpha * data + d_sigma * eps
        else:
            raise ValueError(f"Unknown loss target: {self.loss_target}")

        loss = torch.nn.functional.mse_loss(
            pred.to(torch.float32), target.to(torch.float32), reduction="none"
        )

        if self.weight_fn_name == "min-snr-gamma":
            snr = (alpha / sigma) ** 2
            weights = min_snr_gamma(t, snr)
            # Reshape weights to match batch dim
            weights = weights.view(-1, *([1] * (loss.ndim - 1)))
            loss = loss * weights

        # Sum over batch
        return torch.sum(loss) / B


class EDMLoss(nn.Module):
    """
    Loss function for Karras EDM training.

    It calculates the weighted MSE between the Denoiser output (x_0 estimate) and the ground truth.
    Weighting is automatically derived from the preconditioner to ensure balanced gradients
    for the specific prediction target.
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def forward(self, edm_model: EDMPreconditioner, x: torch.Tensor, prompt=None):
        """
        Args:
            edm_model: An instance of EDMPreconditioner
            x_start: Clean data (y)
        """
        B = x.shape[0]
        device = x.device

        sigma = log_normal_sigma(B, device, self.P_mean, self.P_std)
        view_shape = [-1] + [1] * (x.ndim - 1)
        sigma = sigma.view(*view_shape)

        noise = torch.randn_like(x)
        x_t = x + sigma * noise

        # Get Scalings from the model (for loss weighting)
        # We need c_out to calculate the effective loss weight
        with torch.no_grad():
            _, c_out, _, _ = edm_model.get_scalings(sigma)

        D_x = edm_model(x_t, sigma.view(B))

        # 5. Calculate Loss Weighting
        c_out_safe = torch.where(c_out.abs() < 1e-5, 1e-5 * torch.sign(c_out), c_out)
        weight = 1.0 / (c_out_safe**2)

        # Loss = weight * || D(x+n) - y ||^2
        loss = weight * torch.nn.functional.mse_loss(
            D_x.to(torch.float32), x.to(torch.float32), reduction="none"
        )
        return loss.sum() / B


class ConsistencyTrainingLoss(nn.Module):
    """
    Improved Consistency Training (iCT) Loss.
    Implements:
    - No EMA for teacher (teacher = student with stopgrad).
    - Pseudo-Huber loss.
    - Lognormal noise schedule.
    - Exponential discretization curriculum.
    - Shared dropout masks via RNG state sync.
    """

    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        s0=10,
        s1=1280,
        total_training_steps=400000,
        P_mean=-1.1,
        P_std=2.0,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

        self.s0 = s0
        self.s1 = s1
        self.total_training_steps = total_training_steps

        self.P_mean = P_mean
        self.P_std = P_std

        self.step = 0

        self.conditional = False
        self.model_forward = self._forward_uncond

    def set_conditional(self, conditional):
        self.conditional = conditional
        self.model_forward = self._forward_cond if conditional else self._forward_uncond

    def _forward_cond(self, model, z_t, t, prompt):
        text_cond, attention_mask = model["text_enc"](prompt)
        return model["unet"](
            z_t, t, encoder_hidden_states=text_cond, attention_mask=attention_mask
        )

    def _forward_uncond(self, model, z_t, t, prompt):
        if isinstance(model, (dict, nn.ModuleDict)):
            return model["unet"](z_t, t)
        return model(z_t, t)

    def forward(self, model, x, prompt=None):
        if isinstance(x, (list, tuple)):
            _, data = x
        else:
            data = x

        B = data.shape[0]
        device = data.device
        d = data[0].numel()

        if self.training:
            self.step += 1

        # Discretization Curriculum N(k)
        # K' = floor( K / log2( floor(s1 / s0) + 1 ) )
        val = math.floor(self.s1 / self.s0) + 1
        K_prime = math.floor(self.total_training_steps / math.log2(val))

        # N(k) = min( s0 * 2^(floor(k / K')), s1 ) + 1
        N_k = min(self.s0 * (2 ** math.floor(self.step / K_prime)), self.s1)
        N_k = int(N_k) + 1

        i_grid = torch.arange(N_k, device=device)
        inv_rho = 1 / self.rho
        s_min_inv = self.sigma_min**inv_rho
        s_max_inv = self.sigma_max**inv_rho

        sigma_grid = (
            s_min_inv + i_grid / (N_k - 1) * (s_max_inv - s_min_inv)
        ) ** self.rho

        log_sigma = torch.log(sigma_grid)
        cdf = 0.5 * (
            1 + torch.erf((log_sigma - self.P_mean) / (math.sqrt(2) * self.P_std))
        )
        probs = cdf[1:] - cdf[:-1]
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, num_samples=B, replacement=True)

        view_shape = (B,) + (1,) * (data.ndim - 1)
        t_n = sigma_grid[indices].view(view_shape)
        t_n_plus_1 = sigma_grid[indices + 1].view(view_shape)

        weight = 1.0 / (t_n_plus_1 - t_n).view(B)

        z = torch.randn_like(data)
        x_n_plus_1 = data + t_n_plus_1 * z
        x_n = data + t_n * z

        # Save RNG state to ensure same dropout mask for student and teacher
        rng_state = torch.get_rng_state()
        if data.is_cuda:
            cuda_rng_state = torch.cuda.get_rng_state(device)

        # Online network prediction (Student)
        pred_online = self.model_forward(model, x_n_plus_1, t_n_plus_1.view(B), prompt)

        # Restore RNG state for teacher
        torch.set_rng_state(rng_state)
        if data.is_cuda:
            torch.cuda.set_rng_state(cuda_rng_state, device)

        # Teacher network prediction (EMA removed -> use online with stopgrad)
        with torch.no_grad():
            pred_target = self.model_forward(model, x_n, t_n.view(B), prompt).detach()

        # Pseudo-Huber Loss
        c = 0.00054 * math.sqrt(d)
        diff = pred_online - pred_target

        # ||x - y||_2^2
        sq_l2_norm = (diff**2).flatten(1).sum(dim=1)
        pseudo_huber_loss = torch.sqrt(sq_l2_norm + c**2) - c

        loss = weight * pseudo_huber_loss

        return loss.mean()


class DDGANLoss(nn.Module):
    """
    Computes the Non-Saturating GAN loss for DD-GAN with R1 regularization.
    Returns both Discriminator and Generator losses.
    """

    def __init__(self, schedule, num_timesteps=4, r1_gamma=0.05):
        super().__init__()
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        self.r1_gamma = r1_gamma

    def forward(self, model, x, prompt=None):
        G, D = model["G"], model["D"]
        B, device = x.shape[0], x.device

        t_grid = torch.linspace(0, 1, self.num_timesteps + 1, device=device)
        i = torch.randint(0, self.num_timesteps, (B,), device=device)

        t_curr = t_grid[i]
        t_next = t_grid[i + 1]

        c_alpha, c_sigma, _, _ = self.schedule.get_coefficients(t_curr)
        n_alpha, n_sigma, _, _ = self.schedule.get_coefficients(t_next)

        view_shape = (B,) + (1,) * (x.ndim - 1)
        c_alpha, c_sigma = c_alpha.view(view_shape), c_sigma.view(view_shape)
        n_alpha, n_sigma = n_alpha.view(view_shape), n_sigma.view(view_shape)

        # Sample True Pairs (x_next, x_curr) from forward diffusion
        eps = torch.randn_like(x)
        x_next = n_alpha * x + n_sigma * eps

        alpha_step = c_alpha / n_alpha.clamp(min=1e-5)
        var_step = c_sigma**2 - alpha_step**2 * n_sigma**2
        sigma_step = torch.sqrt(torch.clamp(var_step, min=0.0))

        eps_prime = torch.randn_like(x)
        x_curr = alpha_step * x_next + sigma_step * eps_prime

        x_next = x_next.detach().requires_grad_(True)
        D_real = D(x_next, x_curr.detach(), t_curr)
        errD_real = torch.nn.functional.softplus(-D_real).mean()

        # R1 Penalty
        grad_real = torch.autograd.grad(
            outputs=D_real.sum(), inputs=x_next, create_graph=True
        )[0]
        grad_pen = (grad_real.view(B, -1).norm(2, dim=1) ** 2).mean()
        errD_real = errD_real + self.r1_gamma / 2 * grad_pen

        # Sample Fake Pairs using Generator and Posterior
        z = torch.randn(B, G.latent_dim, device=device)
        x_0_pred = G(x_curr, t_curr, z)

        # q(x_next | x_curr, x_0_pred) mapping
        a_step_sq = (c_alpha**2 / n_alpha**2).clamp(0, 1)
        beta_step = 1 - a_step_sq

        c_x0 = (n_alpha * beta_step) / (c_sigma**2).clamp(min=1e-5)
        c_z = (torch.sqrt(a_step_sq) * n_sigma**2) / (c_sigma**2).clamp(min=1e-5)

        mean = c_x0 * x_0_pred + c_z * x_curr
        var = (n_sigma**2 / (c_sigma**2).clamp(min=1e-5)) * beta_step

        x_next_fake = mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(x_curr)

        D_fake = D(x_next_fake.detach(), x_curr.detach(), t_curr)
        errD_fake = torch.nn.functional.softplus(D_fake).mean()

        loss_D = errD_real + errD_fake

        D_fake_G = D(x_next_fake, x_curr, t_curr)
        loss_G = torch.nn.functional.softplus(-D_fake_G).mean()

        return loss_D, loss_G
