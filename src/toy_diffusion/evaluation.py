import numpy as np
import torch
from scipy.spatial.distance import cdist

try:
    from cleanfid import fid
except ImportError:
    print("Warning: cleanfid not installed. FID computation will fail.")


def compute_fid(fdir1, fdir2, mode="clean", batch_size=64, num_workers=0):
    """
    Computes the FID score between two directories of images using clean-fid.
    This implementation uses the 'clean' resizing mode which is more faithful
    for small resolutions (like 64x64) compared to legacy resize methods.

    Args:
        fdir1 (str): Path to the directory containing real images.
        fdir2 (str): Path to the directory containing generated images.
        mode (str): Resizing mode. Defaults to "clean".
        num_workers (int): Number of workers for data loading.

    Returns:
        float: The computed FID score.
    """
    score = fid.compute_fid(
        fdir1, fdir2, mode=mode, batch_size=batch_size, num_workers=num_workers
    )
    return score


def compute_precision_recall(generated_samples, gt_samples, k=3):
    """
    Computes the Precision and Recall using the k-Nearest Neighbor (k-NN) manifold method.
    This is the standard metric for assessing Generative Models on arbitrary topologies
    (like Pinwheel or GMM) where simple bounding boxes fail.

    Args:
        generated_samples (np.ndarray): Shape (N_gen, D)
        gt_samples (np.ndarray): Shape (N_gt, D) - Ground Truth / Reference data
        k (int): Number of neighbors to estimate the manifold radius.

    Returns:
        precision (float): Fraction of generated samples that fall within the GT manifold. (Quality)
        recall (float): Fraction of GT samples that fall within the generated manifold. (Coverage)
    """
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()
    if isinstance(gt_samples, torch.Tensor):
        gt_samples = gt_samples.detach().cpu().numpy()

    # 1. Estimate Manifolds
    gt_dist = cdist(gt_samples, gt_samples)
    np.fill_diagonal(gt_dist, np.inf)
    # Get distance to k-th neighbor (k-1 index because 0 is 1st)
    gt_radii = np.partition(gt_dist, k - 1, axis=1)[:, k - 1]

    # Calculate distance to the k-th nearest neighbor for each point in Generated
    gen_dist = cdist(generated_samples, generated_samples)
    np.fill_diagonal(gen_dist, np.inf)
    gen_radii = np.partition(gen_dist, k - 1, axis=1)[:, k - 1]

    # 2. Compute Precision (Quality)
    # Distance from each Gen point to all GT points
    d_gen_to_gt = cdist(generated_samples, gt_samples)
    # Check if dist(gen_i, gt_j) <= radius(gt_j) for any j
    in_gt_manifold = np.any(d_gen_to_gt <= gt_radii.reshape(1, -1), axis=1)
    precision = np.mean(in_gt_manifold)

    # 3. Compute Recall (Coverage/Diversity)
    # For each real sample, is it close enough to ANY generated sample?
    d_gt_to_gen = d_gen_to_gt.T
    # Check if dist(gt_i, gen_j) <= radius(gen_j) for any j
    in_gen_manifold = np.any(d_gt_to_gen <= gen_radii.reshape(1, -1), axis=1)
    recall = np.mean(in_gen_manifold)

    return precision, recall


def compute_curvature(trajectories):
    """
    Measures the "Straightness" of the diffusion paths.
    Flow Matching (ODE) aims for straight paths (curvature ~ 1.0).
    DDPM (SDE) often has chaotic/curved paths (curvature >> 1.0).

    Supports both flattened data (B, T, D) and image data (B, T, C, H, W).

    Args:
        trajectories (np.ndarray or torch.Tensor): Shape (B, Steps, D) or (B, Steps, C, H, W)

    Returns:
        mean_curvature (float): Average ratio of Path Length / Displacement.
                                1.0 = Perfectly Straight.
    """
    if isinstance(trajectories, np.ndarray):
        trajectories = torch.from_numpy(trajectories)

    if trajectories.ndim > 3:
        B, T = trajectories.shape[:2]
        trajectories = trajectories.view(B, T, -1)

    # trajectories: (B, T, D)
    # 1. Calculate Displacement (Start to End)
    start = trajectories[:, 0, :]
    end = trajectories[:, -1, :]
    displacement = torch.norm(end - start, dim=-1)

    # 2. Calculate Path Length (Sum of steps)
    # diffs[t] = traj[t+1] - traj[t]
    diffs = trajectories[:, 1:, :] - trajectories[:, :-1, :]
    step_lengths = torch.norm(diffs, dim=-1)
    path_length = step_lengths.sum(dim=-1)

    # 3. Curvature Ratio
    curvature = path_length / (displacement + 1e-8)

    return curvature.mean().item()


def chamfer_distance(set1, set2, is_trajectory=False):
    """
    Computes the Chamfer distance between two sets of data.
    Can handle both point clouds (N, D) and trajectory bundles (N, T, D).

    Args:
        set1: (N1, D) or (N1, T, D)
        set2: (N2, D) or (N2, T, D)
        is_trajectory: If True, treats the entire path as a single feature vector.

    Returns:
        dist (float): The symmetric Chamfer distance.
    """
    if isinstance(set1, torch.Tensor):
        set1 = set1.detach().cpu().numpy()
    if isinstance(set2, torch.Tensor):
        set2 = set2.detach().cpu().numpy()

    if is_trajectory:
        if set1.ndim == 3:
            N, T, D = set1.shape
            set1 = set1.reshape(N, T * D)
        if set2.ndim == 3:
            N, T, D = set2.shape
            set2 = set2.reshape(N, T * D)

    # cdist returns (N1, N2) matrix of distances
    dist_matrix = cdist(set1, set2, metric="euclidean")

    min_dist_1 = np.min(dist_matrix, axis=1)
    min_dist_2 = np.min(dist_matrix, axis=0)

    # Sum of squared distances (Standard Chamfer definition)
    chamfer_dist = np.mean(min_dist_1**2) + np.mean(min_dist_2**2)

    return chamfer_dist
