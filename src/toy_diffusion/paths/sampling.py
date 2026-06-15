import torch
import numpy as np
import math


class CFGModelWrapper:
    """
    Wraps the model to handle Classifier-Free Guidance (CFG) and unconditional forward passes.
    """

    def __init__(
        self,
        model,
        embeddings=None,
        cfg_scale=1.0,
        is_conditional=False,
        attention_mask=None,
    ):
        self.model = model
        self.embeddings = embeddings
        self.cfg_scale = cfg_scale
        self.is_conditional = is_conditional
        self.attention_mask = attention_mask

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.is_conditional and self.cfg_scale > 1.0 and self.embeddings is not None:
            # CFG: Double the batch
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            out = self.model["unet"](
                x_in,
                t_in,
                encoder_hidden_states=self.embeddings,
                attention_mask=self.attention_mask,
            )
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + self.cfg_scale * (out_cond - out_uncond)
        elif self.is_conditional:
            return self.model["unet"](
                x,
                t,
                encoder_hidden_states=self.embeddings,
                attention_mask=self.attention_mask,
            )
        else:
            if isinstance(self.model, (dict, torch.nn.ModuleDict)):
                return self.model["unet"](x, t)
            return self.model(x, t)

    @property
    def inner_model(self):
        return self.model


def convert_prediction_ddpm(
    curr_alpha_scale,
    curr_sigma_scale,
    d_alpha,
    d_sigma,
    pred,
    z,
    prediction_target,
    clip_prediction: bool = True,
):
    curr_alpha_safe = curr_alpha_scale.clamp(min=1e-5)
    curr_sigma_safe = curr_sigma_scale.clamp(min=1e-5)

    if prediction_target == "eps":
        pred_eps = pred
        pred_x = (z - curr_sigma_scale * pred_eps) / curr_alpha_safe
        if clip_prediction:
            pred_x = pred_x.clamp(-1.0, 1.0)
    elif prediction_target == "x":
        pred_x = pred
        pred_eps = (z - curr_alpha_scale * pred_x) / curr_sigma_safe
    elif prediction_target == "v":
        # Invert V to get x0 and eps
        det = curr_alpha_scale * d_sigma - curr_sigma_scale * d_alpha
        det_safe = torch.where(det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det)
        pred_x = (d_sigma * z - curr_sigma_scale * pred) / det_safe
        pred_eps = (curr_alpha_scale * pred - d_alpha * z) / det_safe

        if clip_prediction:
            pred_x = pred_x.clamp(-1.0, 1.0)

    return pred_eps, pred_x, curr_alpha_safe, curr_sigma_safe


@torch.no_grad()
def sample_euler(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    prediction_target="v",
    num_steps=50,
    projection_matrix=None,
    shift=1.0,
    return_traj=False,
    perturb_t: float = None,
    perturb_scale: float = 0.0,
):
    """
    Euler ODE Solver.
    """
    batch_size = x.shape[0]
    device = x.device
    z = x.clone()

    # 2. Time Grid (0 -> 1)
    t_grid = torch.linspace(0, 1, num_steps + 1, device=device)

    # shift > 1 pushes values towards 0 (Noise)
    if shift != 1.0:
        t_grid = t_grid / (shift - (shift - 1) * t_grid)

    dt_steps = t_grid[1:] - t_grid[:-1]

    traj = []
    if return_traj:
        current_z = z.float().cpu().numpy()
        if projection_matrix is not None:
            current_z = current_z @ projection_matrix
        traj.append(current_z)

    perturb_threshold = 0.5 / num_steps

    for i in range(num_steps):
        t_curr = t_grid[i]
        dt = dt_steps[i]

        # We perturb if we are at the requested timestep
        if perturb_t is not None and abs(t_curr.item() - perturb_t) < perturb_threshold:
            # Add Gaussian noise perturbation
            z = z + torch.randn_like(z) * perturb_scale

        t_input = torch.full((batch_size,), t_curr, device=device)

        pred = model_wrapper(z, t_input)

        view_shape = (-1,) + (1,) * (z.ndim - 1)
        alpha, sigma, d_alpha, d_sigma = [
            x.view(view_shape) for x in schedule.get_coefficients(t_input)
        ]

        if prediction_target == "v":
            v = pred

        # using the standard x0
        elif prediction_target == "x":
            # v = d_alpha * x + d_sigma * eps
            # eps = (z - alpha * x) / sigma
            sigma_safe = sigma.clamp(min=1e-5)
            eps_recon = (z - alpha * pred) / sigma_safe
            v = d_alpha * pred + d_sigma * eps_recon

        elif prediction_target == "eps":
            # v = d_alpha * x + d_sigma * eps
            # x = (z - sigma * eps) / alpha
            # Substitute x: v = d_alpha * (z - sigma*eps)/alpha + d_sigma * eps
            # v = (d_alpha/alpha) * z + (d_sigma - d_alpha*sigma/alpha) * eps

            # Stable form for Linear Schedule:
            # alpha=t, sigma=1-t, d_alpha=1, d_sigma=-1
            # v = (z - eps) / t  (singularity at t=0)

            alpha_safe = alpha.clamp(min=1e-5)
            x_recon = (z - sigma * pred) / alpha_safe
            v = d_alpha * x_recon + d_sigma * pred

        # Euler Step
        z = z + v * dt

        if return_traj:
            current_z = z.float().cpu().numpy()
            if projection_matrix is not None:
                current_z = current_z @ projection_matrix
            traj.append(current_z)

    # Project back if necessary
    if projection_matrix is not None:
        # z is (B, D), P is (D, 2)
        final_samples = z.float().cpu().numpy() @ projection_matrix
    else:
        final_samples = z.float().cpu().numpy()

    if return_traj:
        # Stack into (B, Steps, Dim)
        return final_samples, np.stack(traj, axis=1)

    return final_samples


@torch.no_grad()
def sample_ddpm(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    prediction_target="eps",
    num_steps=100,
    projection_matrix=None,
    shift=1.0,
    return_traj=False,
    perturb_t: float = None,
    perturb_scale: float = 0.0,
):
    """
    DDPM SDE Solver using x0.
    """
    batch_size = x.shape[0]
    device = x.device
    z = x.clone()

    # Time Grid: 0 (Noise) -> 1 (Data)
    # Current step t_i (Noisy). Next step t_{i+1} (Cleaner).
    t_grid = torch.linspace(0, 1, num_steps + 1, device=device)

    if shift != 1.0:
        t_grid = t_grid / (shift - (shift - 1) * t_grid)

    traj = []
    if return_traj:
        current_z = z
        if projection_matrix is not None:
            current_z = current_z.cpu().numpy() @ projection_matrix
        traj.append(current_z)

    perturb_threshold = 0.5 / num_steps

    # move from t_grid[i] (Noisy) to t_grid[i+1] (Cleaner)
    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]

        if perturb_t is not None and abs(t_curr.item() - perturb_t) < perturb_threshold:
            z = z + torch.randn_like(z) * perturb_scale

        t_input = torch.full((batch_size,), t_curr, device=device)
        pred = model_wrapper(z, t_input)

        # Note: These are scales (sqrt(alpha_bar)), not variances
        view_shape = (-1,) + (1,) * (z.ndim - 1)
        curr_alpha_scale, curr_sigma_scale, d_alpha, d_sigma = [
            x.view(view_shape) for x in schedule.get_coefficients(t_input)
        ]

        t_next_input = torch.full((batch_size,), t_next, device=device)

        # If we are at the last step (going to t=1), we must enforce the clean data boundary.
        if i == num_steps - 1:
            next_alpha_scale = torch.ones_like(curr_alpha_scale)
            next_sigma_scale = torch.zeros_like(curr_sigma_scale)
        else:
            next_alpha_scale, next_sigma_scale, _, _ = [
                x.view(view_shape) for x in schedule.get_coefficients(t_next_input)
            ]

        curr_alpha_sq = curr_alpha_scale**2
        curr_sigma_sq = curr_sigma_scale**2

        curr_alpha_safe = curr_alpha_scale.clamp(min=1e-5)
        curr_sigma_safe = curr_sigma_scale.clamp(min=1e-5)

        if prediction_target == "eps":
            pred_eps = pred
            # x0 = (z - sigma * eps) / alpha
            pred_x = (z - curr_sigma_scale * pred_eps) / curr_alpha_safe
            pred_x = pred_x.clamp(-1.0, 1.0)

        elif prediction_target == "x":
            pred_x = pred
            # eps = (z - alpha * x0) / sigma
            pred_eps = (z - curr_alpha_scale * pred_x) / curr_sigma_safe
            pred_eps = pred_eps.clamp(-1.0, 1.0)

        elif prediction_target == "v":
            # Invert V to get x0 and eps
            # det = alpha * d_sigma - sigma * d_alpha
            det = curr_alpha_scale * d_sigma - curr_sigma_scale * d_alpha
            det_safe = torch.where(
                det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det
            )

            # x = (d_sigma * z - sigma * v) / det
            pred_x = (d_sigma * z - curr_sigma_scale * pred) / det_safe
            # eps = (alpha * v - d_alpha * z) / det
            pred_eps = (curr_alpha_scale * pred - d_alpha * z) / det_safe

        # alpha_bar_t (current/noisy) = curr_alpha_scale^2
        # alpha_bar_{t-1} (next/cleaner) = next_alpha_scale^2

        # alpha_step = alpha_bar_t / alpha_bar_{t-1}
        alpha_step = (curr_alpha_sq / next_alpha_scale**2).clamp(0, 1)
        beta_step = 1.0 - alpha_step

        # mu = (1 / sqrt(alpha_step)) * (z - (beta_step / sqrt(1 - alpha_bar_t)) * eps)

        # Coeff 1 (x0): (sqrt(alpha_bar_{t-1}) * beta_step) / (1 - alpha_bar_t)
        # Coeff 2 (z):  (sqrt(alpha_step) * (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t)
        # 1 - alpha_bar_t is curr_sigma_sq
        # 1 - alpha_bar_{t-1} is next_sigma_sq

        coeff_x0 = (next_alpha_scale * beta_step) / curr_sigma_sq
        coeff_z = (torch.sqrt(alpha_step) * next_sigma_scale**2) / curr_sigma_sq

        pred_mean = coeff_x0 * pred_x + coeff_z * z

        # sigma_t^2 = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_step
        posterior_variance = (next_sigma_scale**2 / curr_sigma_sq) * beta_step

        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))

        noise = torch.randn_like(z)

        mask = 1.0 if i != (num_steps - 1) else 0.0

        z = pred_mean + mask * (0.5 * posterior_log_variance_clipped).exp() * noise
        if return_traj:
            current_z = z
            if projection_matrix is not None:
                current_z = current_z.cpu().numpy() @ projection_matrix
            traj.append(current_z)

    if projection_matrix is not None:
        final_samples = z.cpu().numpy() @ projection_matrix
    else:
        final_samples = z.cpu().numpy()

    if return_traj:
        return final_samples, np.stack(traj, axis=1)

    return final_samples


@torch.no_grad()
def sample_edm_heun(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    num_steps=35,
    projection_matrix=None,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=1.003,
    return_traj=False,
):
    """
    Karras 2nd Order Heun Sampler (Algorithm 1 + Stochasticity).
    Expects 'model' to be the EDMPreconditioner wrapper.
    """
    batch_size = x.shape[0]
    device = x.device

    # EDM expects initial noise to be scaled by sigma_max
    z = x.clone() * sigma_max

    step_indices = torch.arange(num_steps + 1, device=device, dtype=torch.float32)

    inv_rho = 1.0 / rho
    t_steps = (
        sigma_max**inv_rho
        + step_indices / (num_steps) * (sigma_min**inv_rho - sigma_max**inv_rho)
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    traj = []
    if return_traj:
        current_z = z
        if projection_matrix is not None:
            current_z = current_z.cpu().numpy() @ projection_matrix
        traj.append(current_z)

    for i in range(num_steps):
        sigma_curr = t_steps[i]
        sigma_next = t_steps[i + 1]

        gamma = 0.0
        if S_min <= sigma_curr <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)

        sigma_hat = sigma_curr * (gamma + 1)

        if gamma > 0:
            eps = torch.randn_like(z) * S_noise
            z = z + eps * (sigma_hat**2 - sigma_curr**2).sqrt()

        # Euler Step
        # D_theta predicts "x_start" (clean data)
        # ODE: dx/dt = (x - D(x))/sigma

        denoised = model_wrapper(z, sigma_hat.view(1).repeat(batch_size))
        d_cur = (z - denoised) / sigma_hat

        dt = sigma_next - sigma_hat
        z_next_euler = z + d_cur * dt

        if sigma_next > 1e-6:
            denoised_next = model_wrapper(
                z_next_euler, sigma_next.view(1).repeat(batch_size)
            )
            d_next = (z_next_euler - denoised_next) / sigma_next

            # Average slope
            d_prime = (d_cur + d_next) / 2.0
            z = z + d_prime * dt
        else:
            z = z_next_euler

        if return_traj:
            current_z = z
            if projection_matrix is not None:
                current_z = current_z.cpu().numpy() @ projection_matrix
            traj.append(current_z)

    if projection_matrix is not None:
        final_samples = z.cpu().numpy() @ projection_matrix
    else:
        final_samples = z.cpu().numpy()

    if return_traj:
        return final_samples, np.stack(traj, axis=1)

    return final_samples


@torch.no_grad()
def sample_ddim(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    prediction_target="eps",
    num_steps=100,
    projection_matrix=None,
    shift=1.0,
    eta=0.0,
    return_traj=False,
    perturb_t: float = None,
    perturb_scale: float = 0.0,
    clip_prediction: bool = True,
):
    """
    DDIM / Generalized DDPM Solver.

    Args:
        eta (float): Controls the stochasticity.
                     0.0 = Deterministic DDIM (Probability Flow ODE).
                     1.0 = Standard DDPM (SDE).
    """
    batch_size = x.shape[0]
    device = x.device
    z = x.clone()

    # Time Grid: 0 (Noise) -> 1 (Data)
    t_grid = torch.linspace(0, 1, num_steps + 1, device=device)

    if shift != 1.0:
        t_grid = t_grid / (shift - (shift - 1) * t_grid)

    traj = []
    if return_traj:
        current_z = z.float().cpu().numpy()
        if projection_matrix is not None:
            current_z = current_z @ projection_matrix
        traj.append(current_z)

    perturb_threshold = 0.5 / num_steps

    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]

        if perturb_t is not None and abs(t_curr.item() - perturb_t) < perturb_threshold:
            z = z + torch.randn_like(z) * perturb_scale

        t_input = torch.full((batch_size,), t_curr, device=device)
        pred = model_wrapper(z, t_input)

        view_shape = (-1,) + (1,) * (z.ndim - 1)
        curr_alpha_scale, curr_sigma_scale, d_alpha, d_sigma = [
            x.view(view_shape) for x in schedule.get_coefficients(t_input)
        ]

        t_next_input = torch.full((batch_size,), t_next, device=device)

        if i == num_steps - 1:
            next_alpha_scale = torch.ones_like(curr_alpha_scale)
            next_sigma_scale = torch.zeros_like(curr_sigma_scale)
        else:
            next_alpha_scale, next_sigma_scale, _, _ = [
                x.view(view_shape) for x in schedule.get_coefficients(t_next_input)
            ]

        curr_alpha_safe = curr_alpha_scale.clamp(min=1e-5)
        curr_sigma_safe = curr_sigma_scale.clamp(min=1e-5)

        if prediction_target == "eps":
            pred_eps = pred
            pred_x = (z - curr_sigma_scale * pred_eps) / curr_alpha_safe
            if clip_prediction:
                pred_x = pred_x.clamp(-1.0, 1.0)
        elif prediction_target == "x":
            pred_x = pred
            pred_eps = (z - curr_alpha_scale * pred_x) / curr_sigma_safe
        elif prediction_target == "v":
            # Invert V to get x0 and eps
            det = curr_alpha_scale * d_sigma - curr_sigma_scale * d_alpha
            det_safe = torch.where(
                det.abs() < 1e-5, 1e-5 * torch.sign(det + 1e-35), det
            )
            pred_x = (d_sigma * z - curr_sigma_scale * pred) / det_safe
            pred_eps = (curr_alpha_scale * pred - d_alpha * z) / det_safe

            if clip_prediction:
                pred_x = pred_x.clamp(-1.0, 1.0)

        variance = torch.zeros_like(curr_alpha_scale)
        if i < num_steps - 1:
            curr_alpha_sq = curr_alpha_scale**2
            next_alpha_sq = next_alpha_scale**2
            curr_sigma_sq = curr_sigma_scale**2
            next_sigma_sq = next_sigma_scale**2

            variance = (next_sigma_sq / curr_sigma_sq) * (
                1 - curr_alpha_sq / next_alpha_sq
            )
            variance = variance.clamp(min=0.0)

        std_dev_t = eta * variance.sqrt()

        dir_xt_coeff = (next_sigma_scale**2 - std_dev_t**2).clamp(min=0.0).sqrt()
        pred_sample_direction = dir_xt_coeff * pred_eps

        # x_{t-1} = alpha_next * x_0 + dir_xt + std_dev_t * noise
        z_prev = next_alpha_scale * pred_x + pred_sample_direction

        if eta > 0:
            noise = torch.randn_like(z)
            z_prev = z_prev + std_dev_t * noise

        z = z_prev

        if return_traj:
            current_z = z.float().cpu().numpy()
            if projection_matrix is not None:
                current_z = current_z @ projection_matrix
            traj.append(current_z)

    if projection_matrix is not None:
        final_samples = z.float().cpu().numpy() @ projection_matrix
    else:
        final_samples = z.float().cpu().numpy()

    if return_traj:
        return final_samples, np.stack(traj, axis=1)

    return final_samples


@torch.no_grad()
def sample_dpm_solver_2(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    prediction_target="eps",
    num_steps=100,
    projection_matrix=None,
    shift=1.0,
    return_traj=False,
    perturb_t: float = None,
    perturb_scale: float = 0.0,
    clip_prediction: bool = True,
):
    """
    DPM-Solver++ (2M) Solver.
    A second-order multistep solver for diffusion ODEs.
    Uses 1 NFE per step (reusing the previous step's prediction).
    """
    batch_size = x.shape[0]
    device = x.device
    z = x.clone()

    # Time Grid: 0 (Noise) -> 1 (Data)
    t_grid = torch.linspace(0, 1, num_steps + 1, device=device)

    if shift != 1.0:
        t_grid = t_grid / (shift - (shift - 1) * t_grid)

    traj = []
    if return_traj:
        current_z = z.float().cpu().numpy()
        if projection_matrix is not None:
            current_z = current_z @ projection_matrix
        traj.append(current_z)

    perturb_threshold = 0.5 / num_steps

    # Buffers for multistep (2M)
    prev_pred_x = None
    prev_h = None

    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]

        if perturb_t is not None and abs(t_curr.item() - perturb_t) < perturb_threshold:
            z = z + torch.randn_like(z) * perturb_scale

        t_input = torch.full((batch_size,), t_curr, device=device)
        pred = model_wrapper(z, t_input)

        view_shape = (-1,) + (1,) * (z.ndim - 1)
        curr_alpha_scale, curr_sigma_scale, d_alpha, d_sigma = [
            x.view(view_shape) for x in schedule.get_coefficients(t_input)
        ]
        curr_alpha_safe = curr_alpha_scale.clamp(min=1e-5)
        curr_sigma_safe = curr_sigma_scale.clamp(min=1e-5)
        curr_lambda = torch.log(curr_alpha_safe) - torch.log(curr_sigma_safe)

        t_next_input = torch.full((batch_size,), t_next, device=device)

        next_alpha_scale, next_sigma_scale, next_d_alpha, next_d_sigma = [
            x.view(view_shape) for x in schedule.get_coefficients(t_next_input)
        ]
        next_alpha_safe = next_alpha_scale.clamp(min=1e-5)
        next_sigma_safe = next_sigma_scale.clamp(min=1e-5)
        next_lambda = torch.log(next_alpha_safe) - torch.log(next_sigma_safe)

        # DPM-Solver++ is based on the data prediction model x_theta
        pred_eps, pred_x, _, _ = convert_prediction_ddpm(
            curr_alpha_scale,
            curr_sigma_scale,
            d_alpha,
            d_sigma,
            pred,
            z,
            prediction_target,
            clip_prediction=clip_prediction,
        )

        if i == num_steps - 1:
            z_prev = pred_x
        else:
            # Compute DPM-Solver++ (2M) Update Terms
            h = next_lambda - curr_lambda

            if i == 0:
                # First-order update (like DDIM)
                z_prev = (next_sigma_scale / curr_sigma_safe) * z - next_alpha_scale * (
                    torch.exp(-h) - 1.0
                ) * pred_x
            else:
                # Multistep update (2M)
                r = prev_h / h
                D = (1.0 + 1.0 / (2.0 * r)) * pred_x - (1.0 / (2.0 * r)) * prev_pred_x

                z_prev = (next_sigma_scale / curr_sigma_safe) * z - next_alpha_scale * (
                    torch.exp(-h) - 1.0
                ) * D

        # Update buffers for the next step
        prev_pred_x = pred_x
        prev_h = h

        z = z_prev

        if return_traj:
            current_z = z.float().cpu().numpy()
            if projection_matrix is not None:
                current_z = current_z @ projection_matrix
            traj.append(current_z)

    if projection_matrix is not None:
        final_samples = z.float().cpu().numpy() @ projection_matrix
    else:
        final_samples = z.float().cpu().numpy()

    if return_traj:
        return final_samples, np.stack(traj, axis=1)

    return final_samples


@torch.no_grad()
def sample_consistency_multistep(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    num_steps=1,
    projection_matrix=None,
    return_traj=False,
    sigma_max=80.0,
    sigma_min=0.002,
):
    """
    Multistep Consistency Sampling (Algorithm 1 in the paper).
    Supports 1-step generation or multi-step refinement.
    """
    batch_size = x.shape[0]
    device = x.device

    # Consistency models initial noise is scaled by sigma_max
    z = x.clone() * sigma_max

    # Define the time schedule (tau_1 > tau_2 > ... > tau_{N-1})
    if num_steps == 1:
        t_steps = [sigma_max]
    else:
        # log-spaced schedule from sigma_max to sigma_min
        t_steps = torch.exp(
            torch.linspace(math.log(sigma_max), math.log(sigma_min), num_steps)
        ).tolist()

    # map from noise to data
    x = model_wrapper(z, torch.full((batch_size,), t_steps[0], device=device))

    traj_xt = []
    traj_x0 = []
    if return_traj:
        current_xt = z.float().cpu().numpy()
        current_x0 = x.float().cpu().numpy()
        if projection_matrix is not None:
            current_xt = current_xt @ projection_matrix
            current_x0 = current_x0 @ projection_matrix
        traj_xt.append(current_xt)
        traj_x0.append(current_x0)

    # Multistep refinement
    for i in range(1, num_steps):
        t_curr = t_steps[i]

        # Add noise back
        z_noisy = torch.randn_like(x)
        variance = max(0.0, t_curr**2 - sigma_min**2)
        x_noisy = x + math.sqrt(variance) * z_noisy

        x = model_wrapper(x_noisy, torch.full((batch_size,), t_curr, device=device))

        if return_traj:
            current_xt = x_noisy.float().cpu().numpy()
            current_x0 = x.float().cpu().numpy()
            if projection_matrix is not None:
                current_xt = current_xt @ projection_matrix
                current_x0 = current_x0 @ projection_matrix
            traj_xt.append(current_xt)
            traj_x0.append(current_x0)

    if projection_matrix is not None:
        final_samples = x.float().cpu().numpy() @ projection_matrix
    else:
        final_samples = x.float().cpu().numpy()

    if return_traj:
        return final_samples, (
            np.stack(traj_xt, axis=1),
            np.stack(traj_x0, axis=1),
            t_steps,
        )

    return final_samples


@torch.no_grad()
def sample_ddgan(
    model_wrapper: CFGModelWrapper,
    schedule,
    x: torch.Tensor,
    num_steps=4,
    projection_matrix=None,
    return_traj=False,
):
    """
    Iterative sampling for DD-GAN utilizing the generator and posterior.
    """
    G = model_wrapper.inner_model["G"]
    batch_size = x.shape[0]
    device = x.device
    z_t = x.clone()

    t_grid = torch.linspace(0, 1, num_steps + 1, device=device)

    traj = []
    if return_traj:
        curr_z = z_t.float().cpu().numpy()
        if projection_matrix is not None:
            curr_z = curr_z @ projection_matrix
        traj.append(curr_z)

    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]

        t_in = torch.full((batch_size,), t_curr, device=device)
        z_lat = torch.randn(batch_size, G.latent_dim, device=device)
        x_0_pred = G(z_t, t_in, z_lat)

        c_alpha, c_sigma, _, _ = schedule.get_coefficients(t_in)
        t_next_in = torch.full((batch_size,), t_next, device=device)
        n_alpha, n_sigma, _, _ = schedule.get_coefficients(t_next_in)

        c_alpha, c_sigma = c_alpha.view(-1, 1), c_sigma.view(-1, 1)
        n_alpha, n_sigma = n_alpha.view(-1, 1), n_sigma.view(-1, 1)

        if i == num_steps - 1:
            z_t = x_0_pred
        else:
            a_step_sq = (c_alpha**2 / n_alpha**2).clamp(0, 1)
            beta_step = 1 - a_step_sq

            c_x0 = (n_alpha * beta_step) / (c_sigma**2).clamp(min=1e-5)
            c_z = (torch.sqrt(a_step_sq) * n_sigma**2) / (c_sigma**2).clamp(min=1e-5)

            mean = c_x0 * x_0_pred + c_z * z_t
            var = (n_sigma**2 / (c_sigma**2).clamp(min=1e-5)) * beta_step

            z_t = mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(z_t)

        if return_traj:
            curr_z = z_t.float().cpu().numpy()
            if projection_matrix is not None:
                curr_z = curr_z @ projection_matrix
            traj.append(curr_z)

    if projection_matrix is not None:
        final_samples = z_t.float().cpu().numpy() @ projection_matrix
    else:
        final_samples = z_t.float().cpu().numpy()

    if return_traj:
        return final_samples, np.stack(traj, axis=1)
    return final_samples


def generate_samples(
    model,
    schedule,
    batch_size: int = 1000,
    data_shape=(2,),
    x: torch.Tensor = None,
    diffusion_type: str = "linear",
    prediction_target: str = "v",
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    is_conditional: bool = False,
    projection_matrix=None,
    return_traj: bool = False,
    vae: torch.nn.Module = None,
    device: str = "cuda",
    vae_scale: float = 1.0,
    vae_shift: float = 0.0,
    vae_batch_size: int = 32,
    **kwargs,
):
    """
    Main entry point for generating samples.
    Wraps the model for CFG (if applicable) and calls the correct sampler.
    """
    # --- Prepare Noise ---
    if x is None:
        shape = (batch_size, *data_shape)
        x = torch.randn(shape, device=device)

    # 2. Model Wrapper
    model_wrapper = CFGModelWrapper(
        model,
        embeddings=embeddings,
        cfg_scale=cfg_scale,
        is_conditional=is_conditional,
        attention_mask=attention_mask,
    )
    # parse the kwargs as each func has own signature
    extra_kwargs = {}
    extra_kwargs["perturb_t"] = kwargs.get("perturb_t", None)
    extra_kwargs["perturb_scale"] = kwargs.get("perturb_scale", 0.0)
    extra_kwargs["shift"] = kwargs.get("shift", 1.0)
    # 3. Call specific sampler
    if diffusion_type == "linear":
        samples = sample_euler(
            model_wrapper,
            schedule,
            x,
            prediction_target,
            num_steps,
            projection_matrix=projection_matrix,
            return_traj=return_traj,
            **extra_kwargs,
        )
    elif diffusion_type == "ddpm":
        # no clip for latent
        extra_kwargs["clip_prediction"] = kwargs.get("clip_prediction", False)
        sampler_type = kwargs.get("sampler_type", "ddim")
        if sampler_type == "dpm-solver":
            samples = sample_dpm_solver_2(
                model_wrapper,
                schedule,
                x,
                prediction_target,
                num_steps,
                projection_matrix=projection_matrix,
                return_traj=return_traj,
                **extra_kwargs,
            )
        else:
            samples = sample_ddim(
                model_wrapper,
                schedule,
                x,
                prediction_target,
                num_steps,
                projection_matrix=projection_matrix,
                return_traj=return_traj,
                **extra_kwargs,
            )
    elif diffusion_type == "karras_edm":
        samples = sample_edm_heun(
            model_wrapper,
            schedule,
            x,
            num_steps=num_steps,
            projection_matrix=projection_matrix,
            return_traj=return_traj,
            **extra_kwargs,
        )
    elif diffusion_type == "consistency":
        samples = sample_consistency_multistep(
            model_wrapper,
            schedule,
            x,
            num_steps=kwargs.get("cm_steps", 4),
            projection_matrix=projection_matrix,
            return_traj=return_traj,
        )
    elif diffusion_type == "ddgan":
        samples = sample_ddgan(
            model_wrapper,
            schedule,
            x,
            num_steps=kwargs.get("ddgan_steps", 4),
            projection_matrix=projection_matrix,
            return_traj=return_traj,
        )
    else:
        raise ValueError(f"Unknown diffusion type {diffusion_type}")

    # VAE Decoding
    if vae is not None:
        final_z = samples[0] if return_traj else samples
        final_z_tensor = (
            torch.tensor(final_z, device=x.device).float()
            if isinstance(final_z, np.ndarray)
            else final_z.float()
        )

        final_z_tensor = final_z_tensor / vae_scale + vae_shift
        # Batched decoding to prevent OOM
        decoded_list = []
        with torch.no_grad():
            for i in range(0, final_z_tensor.shape[0], vae_batch_size):
                batch_z = final_z_tensor[i : i + vae_batch_size]
                decoded_batch = vae.decode(batch_z).sample
                decoded_batch = decoded_batch.float().clamp(-1, 1).cpu()
                decoded_list.append(decoded_batch)

        decoded = torch.cat(decoded_list, dim=0)
        decoded_np = decoded.permute(0, 2, 3, 1).numpy()

        del final_z_tensor
        del decoded
        del decoded_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_traj:
            traj_z = samples[1]

            Steps, B, C, H_lat, W_lat = traj_z.shape
            decoded_steps = []

            for t in range(Steps):
                latent_step = torch.from_numpy(traj_z[t]).to(x.device).float()
                latent_step = latent_step / vae_scale + vae_shift

                step_decoded_list = []
                with torch.no_grad():
                    for i in range(0, B, vae_batch_size):
                        batch_z = latent_step[i : i + vae_batch_size]
                        decoded_batch = vae.decode(batch_z).sample
                        decoded_batch = decoded_batch.float().clamp(-1, 1).cpu()
                        step_decoded_list.append(decoded_batch)

                decoded_step = torch.cat(step_decoded_list, dim=0)
                decoded_steps.append(decoded_step)
                del latent_step
                del decoded_step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Stack into (Steps, B, C, H, W)
            decoded_traj = torch.stack(decoded_steps)

            # Permute to (B, Steps, C, H, W) for the visualization function
            decoded_traj_np = decoded_traj.permute(1, 0, 2, 3, 4).numpy()

            return decoded_np, decoded_traj_np

        return decoded_np

    return samples
