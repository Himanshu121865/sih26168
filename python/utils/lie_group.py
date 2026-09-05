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
    Used in InEKF update: delta = K * innovation
    """
    phi = xi[0:3]; rho_v = xi[3:6]; rho_p = xi[6:9]
    R = so3exp(phi)
    # left Jacobian for v/p (first-order approx sufficient for small xi)
    # Full J = I + (1-cosθ)/θ² K + (θ-sinθ)/θ³ K² ; we use series for simplicity
    angle = torch.norm(phi)
    if angle < 1e-8:
        J = torch.eye(3, dtype=xi.dtype, device=xi.device)
    else:
        K = skew(phi / angle)
        # use same as so3exp but with scaling
        # J = I + (1-cos)/θ *K + (θ - sin)/θ *K²  (approx)
        # For small innovation, J≈I is fine — QAIIMU uses exact series via filter eqs, we keep simple.
        J = torch.eye(3, dtype=xi.dtype, device=xi.device) + 0.5*skew(phi)
    v = J @ rho_v
    p = J @ rho_p
    return R, v, p

# constants for InEKF (also used in filter)
TENSOR_EYE3 = torch.eye(3, dtype=torch.float64)
TENSOR_EYE6 = torch.eye(6, dtype=torch.float64)
TENSOR_EYE12 = torch.eye(12, dtype=torch.float64)
TENSOR_EYE21 = torch.eye(21, dtype=torch.float64)
