"""
metrics.py — P0 #5 fix (from harsh/eval/metrics.py:49-177)
ATE, RTE, drift%, coverage with SE2 Umeyama alignment.

Usage:
  from python.eval.metrics import ate, rte, drift_pct, coverage
"""
import numpy as np

def _umeyama_se2(src, dst):
    """
    src, dst (N,2) -> R(2,2), t(2,), scale (unused, 1.0)
    Computes SE(2) alignment via Umeyama.
    """
    assert src.shape == dst.shape and src.shape[1] == 2
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1, d])
    R = U @ D @ Vt
    t = dst_mean - R @ src_mean
    return R, t

def ate(est_xy, gt_xy, align=True):
    """
    Absolute Trajectory Error RMSE
    est_xy, gt_xy (N,2) -> RMSE after SE2 alignment if align
    """
    if align:
        R, t = _umeyama_se2(est_xy, gt_xy)
        est_aligned = (R @ est_xy.T).T + t
    else:
        est_aligned = est_xy
    err = np.linalg.norm(est_aligned - gt_xy, axis=1)
    return float(np.sqrt((err**2).mean())), err

def rte(est_xy, gt_xy, t_s, window_s=60.0):
    """
    Relative Trajectory Error over window_s
    est_xy, gt_xy (N,2), t_s (N,) seconds
    Returns mean RPE over sliding window
    """
    errs = []
    for i in range(len(t_s)):
        t_end = t_s[i] + window_s
        j = np.searchsorted(t_s, t_end)
        if j >= len(t_s) or j <= i:
            continue
        delta_est = est_xy[j] - est_xy[i]
        delta_gt = gt_xy[j] - gt_xy[i]
        errs.append(np.linalg.norm(delta_est - delta_gt))
    return float(np.mean(errs)) if errs else 0.0

def drift_pct(final_error_m, total_distance_m):
    """Drift % = final_error / total_distance *100"""
    if total_distance_m < 1e-6:
        return 0.0
    return final_error_m / total_distance_m * 100

def coverage(err_xyz, sigma_xyz, k=1.0):
    """
    Calibration coverage: mean(|err| <= k*sigma)
    err_xyz (N,3) or (N,), sigma_xyz (N,3) or (N,), k=1 → 68% expected
    """
    if err_xyz.ndim == 1:
        err_xyz = err_xyz[:, None]
        sigma_xyz = sigma_xyz[:, None] if sigma_xyz.ndim==1 else sigma_xyz
    # per-axis
    within = np.abs(err_xyz) <= k * sigma_xyz
    return float(within.mean(axis=0).mean()) if within.size else 0.0

def total_distance(gt_xy):
    """Total path length sum(||delta||)"""
    deltas = np.diff(gt_xy, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())

# test
if __name__ == "__main__":
    np.random.seed(0)
    gt = np.cumsum(np.random.randn(100,2)*0.1, axis=0)
    est = gt + np.random.randn(100,2)*0.05
    ate_rmse, _ = ate(est, gt)
    print(f"ATE {ate_rmse:.3f}")
    print(f"RTE 60s {rte(est, gt, np.arange(100)*0.1):.3f}")
    print(f"drift {drift_pct(np.linalg.norm(est[-1]-gt[-1]), total_distance(gt)):.2f}%")
    print(f"coverage {coverage(np.random.randn(100,1), np.ones((100,1))):.2f} (expect ~0.68)")
