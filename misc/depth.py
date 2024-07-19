import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange


def estimate_depth_scale_kitti(depth, depth_gt):
    """
    depth: [1, 1, H, W]
    depth_gt: [N, 2]
    """
    eps = 1e-7
    depth = rearrange(depth, "1 1 h w -> (h w)")
    depth_gt = rearrange(depth_gt, "1 1 h w -> (h w)")
    valid_depth = depth_gt != 0
    depth = depth[valid_depth]
    depth_gt = depth_gt[valid_depth]

    scale = (depth.log() - depth_gt.log()).mean().exp()
    return scale


def estimate_depth_scale(depth, sparse_depth):
    """
    depth: [1, 1, H, W]
    sparse_depth: [N, 3]
    """
    eps = 1e-7
    device = depth.device
    sparse_depth = sparse_depth.to(device)
    if sparse_depth.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)
    xy = sparse_depth[:, :2]
    z = sparse_depth[:, 2]
    xy = xy.unsqueeze(0).unsqueeze(0)
    depth_pred = F.grid_sample(depth, xy.to(depth.device), align_corners=False)
    depth_pred = depth_pred.squeeze()
    # z = torch.max(z, torch.tensor(eps, dtype=z.dtype, device=z.device))
    good_depth = torch.logical_and(z > eps, depth_pred > eps)
    z = z[good_depth]
    depth_pred = depth_pred[good_depth]

    if z.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    scale = (depth_pred.log() - z.log()).mean().exp()
    return scale


def estimate_depth_scale_ransac(depth, sparse_depth, num_iterations=1000, sample_size=5, threshold=0.1):
    best_scale = None
    best_inliers = 0

    device = depth.device
    sparse_depth = sparse_depth.to(device)

    xy = sparse_depth[:, :2]
    z = sparse_depth[:, 2]
    xy = xy.unsqueeze(0).unsqueeze(0)
    depth_pred = F.grid_sample(depth, xy.to(depth.device), align_corners=False)
    depth_pred = depth_pred.squeeze()
    eps=1e-7
    # z = torch.max(z, torch.tensor(eps, dtype=z.dtype, device=z.device))
    good_depth = torch.logical_and(z > eps, depth_pred > eps)

    if good_depth.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)
    z = z[good_depth]
    depth_pred = depth_pred[good_depth]

    if z.shape[0] < 10:
        return torch.tensor(1.0, device=device, dtype=torch.float32)

    if z.shape[0] <= sample_size:
        return (depth_pred.log() - z.log()).mean().exp()

    for _ in range(num_iterations):
        # Step 1: Random Sample Selection
        sample_indices = random.sample(range(z.shape[0]), sample_size)
        # Step 2: Estimation of Scale
        scale = (depth_pred[sample_indices].log() - z[sample_indices].log()).mean().exp()

        # Step 3: Inlier Detection
        inliers = torch.abs(depth_pred.log() - (z*scale).log()) < threshold

        # Step 5: Consensus Set Selection
        num_inliers = torch.sum(inliers)
        if num_inliers > best_inliers:
            best_scale = scale
            best_inliers = num_inliers
    if best_scale is None:
        return (depth_pred.log() - z.log()).mean().exp()
    return best_scale

CMAP_DEFAULT = 'plasma'
def gray2rgb(im, cmap=CMAP_DEFAULT):
    cmap = plt.get_cmap(cmap)
    result_img = cmap(im.astype(np.float32))
    if result_img.shape[2] > 3:
        result_img = np.delete(result_img, 3, 2)
    return result_img


def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None, cmap=CMAP_DEFAULT,
                                return_normalizer=False):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.

    depth = np.squeeze(depth)

    depth_f = depth.flatten()
    depth_f = depth_f[depth_f != 0]
    disp_f = 1.0 / (depth_f + 1e-6)
    percentile = np.percentile(disp_f, pc)

    disp = 1.0 / (depth + 1e-6)
    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (percentile + 1e-6)
    disp = np.clip(disp, 0, 1)
    disp = gray2rgb(disp, cmap=cmap)
    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]
    if return_normalizer:
        return disp, percentile + 1e-6
    return disp