"""
lie_group.py — copied from ref/QAIIMU/utils/lie_group_utils.py (Step 3.5)
Provides so3exp, skew, sen3exp for InEKF (SE2(3)).
Kept float64 as in reference (paper uses float64 in filter, float32 in CNN).
"""
import torch

def skew(w: torch.Tensor) -> torch.Tensor:
    """w (3,) -> [w]_x (3,3)"""
    wx = torch.zeros(3,3, dtype=w.dtype, device=w.device)
    wx[0,1] = -w[2]; wx[0,2] =  w[1]
    wx[1,0] =  w[2]; wx[1,2] = -w[0]
    wx[2,0] = -w[1]; wx[2,1] =  w[0]
    return wx

def so3exp(phi: torch.Tensor) -> torch.Tensor:
    """exp([phi]_x) via Rodrigues (phi 3,) -> R 3x3"""
    angle = torch.norm(phi)
    if angle < 1e-8:
        return torch.eye(3, dtype=phi.dtype, device=phi.device) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return torch.eye(3, dtype=phi.dtype, device=phi.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K @ K)

def sen3exp(xi: torch.Tensor):
    """
    SE2(3) exp for xi 9D = [phi(3), rho_v(3), rho_p(3)] -> (R, v, p)
    Exact left Jacobian (parity with python/inekf_harness.py:70-88 and
    ref/QAIIMU filter_update_improved). Single source of truth — harness
    imports from here.
    """
    phi = xi[0:3]
    ang = torch.norm(phi)
    K = skew(phi)
    dtype, device = xi.dtype, xi.device
    if ang < 1e-10:
        J = torch.eye(3, dtype=dtype, device=device) + 0.5 * K + (1.0 / 6.0) * (K @ K)
        R = torch.eye(3, dtype=dtype, device=device) + K + 0.5 * (K @ K)
    else:
        ang2 = ang * ang
        s, c = torch.sin(ang), torch.cos(ang)
        eye = torch.eye(3, dtype=dtype, device=device)
        phi_col = phi.unsqueeze(1)
        oo = (phi_col @ phi_col.t()) / ang2
        J = (s / ang) * eye + (1 - s / ang) * oo + ((1 - c) / ang) * K
        R = c * eye + (1 - c) * oo + s * K
    v = J @ xi[3:6]
    p = J @ xi[6:9]
    return R, v, p

# constants for InEKF (also used in filter)
TENSOR_EYE3 = torch.eye(3, dtype=torch.float64)
TENSOR_EYE6 = torch.eye(6, dtype=torch.float64)
TENSOR_EYE12 = torch.eye(12, dtype=torch.float64)
TENSOR_EYE21 = torch.eye(21, dtype=torch.float64)
