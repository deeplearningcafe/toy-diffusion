import torch
import numpy as np
from toy_diffusion.paths.sampling import DDPMSchedule


def test_derivations():
    print("--- Testing Derivations for Flow Matching vs DDPM ---")

    B, C, H, W = 2, 4, 32, 32
    x_start = torch.randn(B, C, H, W)
    eps = torch.randn(B, C, H, W)
    t = torch.rand(B)

    def view_t(val):
        return val.view(-1, 1, 1, 1)

    # Test Case A: Linear Schedule (Rectified Flow)
    # alpha = t, sigma = 1-t
    print("\n[Test Case A: Rectified Flow / Linear Schedule]")
    alpha = t
    sigma = 1 - t
    d_alpha = torch.ones_like(t)
    d_sigma = -torch.ones_like(t)

    z = view_t(alpha) * x_start + view_t(sigma) * eps
    v_gt = view_t(d_alpha) * x_start + view_t(d_sigma) * eps  # v = x - eps

    v_simple = x_start - eps
    diff_simple = (v_gt - v_simple).abs().max()
    print(f"  > Verify v = x - eps: Max Diff = {diff_simple:.1e} (Should be 0)")

    det = alpha * d_sigma - sigma * d_alpha
    det = view_t(det)

    # Predict x and eps from v_gt
    x_rec = (view_t(d_sigma) * z - view_t(sigma) * v_gt) / det
    eps_rec = (view_t(alpha) * v_gt - view_t(d_alpha) * z) / det

    err_x = (x_rec - x_start).abs().max()
    err_eps = (eps_rec - eps).abs().max()

    print(f"  > Reconstruction Error x:   {err_x:.1e}")
    print(f"  > Reconstruction Error eps: {err_eps:.1e}")

    if err_x < 1e-5 and err_eps < 1e-5:
        print("  > [PASS] Linear Schedule Inversion is correct.")
    else:
        print("  > [FAIL] Linear Schedule Inversion.")

    # Test Case B: DDPM / Cosine Schedule (Variance Preserving)
    # alpha = cos(t * pi/2), sigma = sin(t * pi/2)
    # Note: This mimics a VP schedule where alpha^2 + sigma^2 = 1
    print("\n[Test Case B: DDPM / Cosine Schedule]")

    pi = torch.pi
    alpha_vp = torch.cos(t * pi / 2)
    sigma_vp = torch.sin(t * pi / 2)
    d_alpha_vp = -(pi / 2) * torch.sin(t * pi / 2)
    d_sigma_vp = (pi / 2) * torch.cos(t * pi / 2)

    z_vp = view_t(alpha_vp) * x_start + view_t(sigma_vp) * eps
    v_gt_vp = view_t(d_alpha_vp) * x_start + view_t(d_sigma_vp) * eps

    # Check SNR relationship: v should be related to eps
    # v = (d_alpha/alpha) * z + (d_sigma - d_alpha*sigma/alpha) * eps
    term1 = (view_t(d_alpha_vp) / view_t(alpha_vp)) * z_vp
    term2 = (
        view_t(d_sigma_vp) - view_t(d_alpha_vp) * view_t(sigma_vp) / view_t(alpha_vp)
    ) * eps
    v_derived = term1 + term2

    mask = t < 0.99
    if mask.sum() > 0:
        diff_derived = (v_gt_vp[mask] - v_derived[mask]).abs().max()
        print(f"  > Verify v vs eps relation: Max Diff = {diff_derived:.1e}")

    det_vp = alpha_vp * d_sigma_vp - sigma_vp * d_alpha_vp
    det_vp = view_t(det_vp)

    # Sanity check determinant for Cosine schedule
    # det = cos * (pi/2 cos) - sin * (-pi/2 sin) = pi/2 * (cos^2 + sin^2) = pi/2
    print(f"  > Determinant Mean (Should be pi/2 approx 1.57): {det_vp.mean():.4f}")

    x_rec_vp = (view_t(d_sigma_vp) * z_vp - view_t(sigma_vp) * v_gt_vp) / det_vp
    eps_rec_vp = (view_t(alpha_vp) * v_gt_vp - view_t(d_alpha_vp) * z_vp) / det_vp

    err_x_vp = (x_rec_vp - x_start).abs().max()
    err_eps_vp = (eps_rec_vp - eps).abs().max()

    print(f"  > Reconstruction Error x:   {err_x_vp:.1e}")
    print(f"  > Reconstruction Error eps: {err_eps_vp:.1e}")

    if err_x_vp < 1e-5:
        print("  > [PASS] DDPM Schedule Inversion is correct.")
    else:
        print("  > [FAIL] DDPM Schedule Inversion.")

    # Test Case C: SD 1.5 Linear Beta Schedule (Discrete -> Continuous)
    print("\n[Test Case C: SD 1.5 Linear Beta Schedule]")

    ddpm_schedule = DDPMSchedule(device=x_start.device)

    alpha_d, sigma_d, d_alpha_d, d_sigma_d = ddpm_schedule.get_coefficients(t)

    z_d = view_t(alpha_d) * x_start + view_t(sigma_d) * eps

    # v = d_alpha * x + d_sigma * eps (Definition of velocity for this path)
    v_gt_d = view_t(d_alpha_d) * x_start + view_t(d_sigma_d) * eps

    # 4. Inversion Logic (Solving the linear system)
    # det = alpha * d_sigma - sigma * d_alpha
    det_d = alpha_d * d_sigma_d - sigma_d * d_alpha_d
    det_d = view_t(det_d)

    # Check Determinant Stability
    # Since alpha increases (d_alpha > 0) and sigma decreases (d_sigma < 0),
    # det should be strictly negative.
    print(f"  > Determinant Mean: {det_d.mean():.4f} (Should be negative)")
    if (det_d.abs() < 1e-4).any():
        print(
            "  > [WARNING] Determinant is very close to zero for some t. Instability possible."
        )

    # Apply Cramer's Rule / Inversion
    # x = (d_sigma * z - sigma * v) / det
    x_rec_d = (view_t(d_sigma_d) * z_d - view_t(sigma_d) * v_gt_d) / det_d

    # eps = (alpha * v - d_alpha * z) / det
    eps_rec_d = (view_t(alpha_d) * v_gt_d - view_t(d_alpha_d) * z_d) / det_d

    # 5. Measure Error
    # Note: Error will be higher here than in Test B because DDPMSchedule uses
    # linear interpolation and finite differences (approximate derivatives).
    err_x_d = (x_rec_d - x_start).abs().max()
    err_eps_d = (eps_rec_d - eps).abs().max()

    print(f"  > Reconstruction Error x:   {err_x_d:.1e}")
    print(f"  > Reconstruction Error eps: {err_eps_d:.1e}")

    # Tolerance is looser (1e-3 to 1e-4) due to finite difference approximation
    if err_x_d < 1e-4:
        print("  > [PASS] SD 1.5 Schedule Inversion is correct (within approx limits).")
    else:
        print("  > [FAIL] SD 1.5 Schedule Inversion error is too high.")


if __name__ == "__main__":
    test_derivations()
