"""Trajectory error metrics (from harsh/eval/metrics.py:49-177).

ATE, RTE, drift% and calibration coverage with SE(2) Umeyama alignment.

Usage:
    from python.eval.metrics import ate, rte, drift_pct, coverage
"""

from __future__ import annotations

import numpy as np


def _umeyama_se2(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the SE(2) transform aligning ``src`` to ``dst``.

    Args:
        src: Source trajectory, shape (N, 2).
        dst: Destination trajectory, shape (N, 2).

    Returns:
        Tuple of (R, t) with rotation shape (2, 2) and translation shape (2,).

    Raises:
        ValueError: If shapes are not matching (N, 2) pairs.
    """
    if src.shape != dst.shape or src.shape[1] != 2:
        raise ValueError(f"expected matching (N,2) pairs, got {src.shape} vs {dst.shape}")
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / n
    U, _S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1, d])
    R = U @ D @ Vt
    t = dst_mean - R @ src_mean
    return R, t


def ate(
    est_xy: np.ndarray, gt_xy: np.ndarray, align: bool = True
) -> tuple[float, np.ndarray]:
    """Compute absolute trajectory error (RMSE).

    Args:
        est_xy: Estimated trajectory, shape (N, 2).
        gt_xy: Ground-truth trajectory, shape (N, 2).
        align: If True, align with SE(2) Umeyama before scoring.

    Returns:
        Tuple of (RMSE, per-sample error vector of length N).
    """
    if align:
        R, t = _umeyama_se2(est_xy, gt_xy)
        est_aligned = (R @ est_xy.T).T + t
    else:
        est_aligned = est_xy
    err = np.linalg.norm(est_aligned - gt_xy, axis=1)
    return float(np.sqrt((err**2).mean())), err


def rte(
    est_xy: np.ndarray, gt_xy: np.ndarray, t_s: np.ndarray, window_s: float = 60.0
) -> float:
    """Compute mean relative pose error over a sliding time window.

    Args:
        est_xy: Estimated trajectory, shape (N, 2).
        gt_xy: Ground-truth trajectory, shape (N, 2).
        t_s: Timestamps in seconds, shape (N,).
        window_s: Sliding window length in seconds.

    Returns:
        Mean displacement error over all valid windows (0.0 if none).
    """
    errs = []
    for i in range(len(t_s)):
        t_end = t_s[i] + window_s
        j = int(np.searchsorted(t_s, t_end))
        if j >= len(t_s) or j <= i:
            continue
        delta_est = est_xy[j] - est_xy[i]
        delta_gt = gt_xy[j] - gt_xy[i]
        errs.append(np.linalg.norm(delta_est - delta_gt))
    return float(np.mean(errs)) if errs else 0.0


def drift_pct(final_error_m: float, total_distance_m: float) -> float:
    """Compute drift as a percentage of distance travelled.

    Args:
        final_error_m: Final position error in meters.
        total_distance_m: Total ground-truth path length in meters.

    Returns:
        ``final_error / total_distance * 100`` (0.0 for degenerate paths).
    """
    if total_distance_m < 1e-6:
        return 0.0
    return final_error_m / total_distance_m * 100


def coverage(err_xyz: np.ndarray, sigma_xyz: np.ndarray, k: float = 1.0) -> float:
    """Compute calibration coverage ``mean(|err| <= k*sigma)``.

    Args:
        err_xyz: Errors, shape (N,) or (N, C).
        sigma_xyz: Predicted stds, shape (N,) or (N, C).
        k: Sigma multiplier (k=1 expects ~0.68 for Gaussians).

    Returns:
        Fraction of entries within the bound.
    """
    if err_xyz.ndim == 1:
        err_xyz = err_xyz[:, None]
        sigma_xyz = sigma_xyz[:, None] if sigma_xyz.ndim == 1 else sigma_xyz
    # per-axis
    within = np.abs(err_xyz) <= k * sigma_xyz
    return float(within.mean(axis=0).mean()) if within.size else 0.0


def total_distance(gt_xy: np.ndarray) -> float:
    """Compute total path length.

    Args:
        gt_xy: Trajectory, shape (N, 2).

    Returns:
        Sum of segment lengths in the same units as the input.
    """
    deltas = np.diff(gt_xy, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


# test
if __name__ == "__main__":
    np.random.seed(0)
    gt = np.cumsum(np.random.randn(100, 2) * 0.1, axis=0)
    est = gt + np.random.randn(100, 2) * 0.05
    ate_rmse, _ = ate(est, gt)
    print(f"ATE {ate_rmse:.3f}")
    print(f"RTE 60s {rte(est, gt, np.arange(100) * 0.1):.3f}")
    print(f"drift {drift_pct(np.linalg.norm(est[-1] - gt[-1]), total_distance(gt)):.2f}%")
    print(f"coverage {coverage(np.random.randn(100, 1), np.ones((100, 1))):.2f} (expect ~0.68)")
