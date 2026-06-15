import torch
import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to sys.path
sys.path.append(parent_dir)
from sampling import DDPMSchedule


def test_variance_preserving_property():
    print("\n--- Testing DDPM Schedule Variance Preservation ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_train_timesteps = 1000
    schedule = DDPMSchedule(device=device, num_train_timesteps=num_train_timesteps)

    t = torch.linspace(0, 1, num_train_timesteps, device=device)

    alpha, sigma, d_alpha, d_sigma = schedule.get_coefficients(t)

    # Check 1: Total Variance (alpha^2 + sigma^2)
    total_var = alpha**2 + sigma**2
    mean_var = total_var.mean().item()
    min_var = total_var.min().item()
    max_var = total_var.max().item()

    print(f"Total Variance (alpha^2 + sigma^2):")
    print(f"  > Mean: {mean_var:.6f}")
    print(f"  > Min:  {min_var:.6f} (Likely between steps)")
    print(f"  > Max:  {max_var:.6f} (Likely on steps)")

    is_strict_vp = torch.allclose(total_var, torch.ones_like(total_var), atol=1e-7)
    print(f"  > Strictly VP (tol=1e-7)? {is_strict_vp}")

    if not is_strict_vp:
        print("    [NOTE] The schedule is NOT strictly VP due to linear interpolation.")
        print("    Linear interpolation cuts through the unit circle arc.")
        print(f"    Max Deviation: {1.0 - min_var:.2e}")

    # Check 2: Derivative Consistency
    # If alpha^2 + sigma^2 = 1, differentiating w.r.t t gives:
    # 2*alpha*d_alpha + 2*sigma*d_sigma = 0
    # => alpha*d_alpha + sigma*d_sigma = 0
    dot_prod = alpha * d_alpha + sigma * d_sigma
    max_dot_error = dot_prod.abs().max().item()

    print(f"\nDerivative Consistency (alpha*d_alpha + sigma*d_sigma = 0):")
    print(f"  > Max Deviation: {max_dot_error:.2e}")

    if max_dot_error > 1e-3:
        print(
            "  > [WARNING] Derivatives are significantly inconsistent with VP constraint."
        )
        print("    This might affect Velocity (v) calculation accuracy.")
    else:
        print("  > [PASS] Derivatives are consistent enough for training.")

    # Check 3: Boundary Conditions
    # t=0 -> Noise (alpha=0, sigma=1) ??
    t_bounds = torch.tensor([0.0, 1.0], device=device)
    a_b, s_b, _, _ = schedule.get_coefficients(t_bounds)

    print(f"\nBoundary Conditions:")
    print(f"  > t=0 (Noise?): alpha={a_b[0]:.4f}, sigma={s_b[0]:.4f}")
    print(f"  > t=1 (Data?):  alpha={a_b[1]:.4f}, sigma={s_b[1]:.4f}")

    # In SD1.5:
    # alpha at T (Noise) is usually very small but not 0.
    # sigma at T is very close to 1.
    if s_b[0] > 0.99:
        print("  > [PASS] t=0 corresponds to high noise.")
    else:
        print("  > [FAIL] t=0 does not look like pure noise.")


if __name__ == "__main__":
    test_variance_preserving_property()
