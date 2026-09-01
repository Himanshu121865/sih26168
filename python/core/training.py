"""core/training.py — Shared train loop + NLL + yaw augment (extracted from train_avnet.py)."""
import math, random
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import numpy as np; np.random.seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def augment_random_yaw(x: torch.Tensor):
    """Rotate horizontal acc x,y (0,1) and gyro pitch/roll (4,5). x (B,200,6)"""
    B=x.shape[0]
    thetas=torch.rand(B, device=x.device)*2*math.pi
    cos_t=torch.cos(thetas); sin_t=torch.sin(thetas)
    xa=x.clone()
    ax=x[:,:,0]; ay=x[:,:,1]
    xa[:,:,0]=ax*cos_t.unsqueeze(-1) - ay*sin_t.unsqueeze(-1)
    xa[:,:,1]=ax*sin_t.unsqueeze(-1) + ay*cos_t.unsqueeze(-1)
    gx=x[:,:,5]; gy=x[:,:,4]
    xa[:,:,5]=gx*cos_t.unsqueeze(-1) - gy*sin_t.unsqueeze(-1)
    xa[:,:,4]=gx*sin_t.unsqueeze(-1) + gy*cos_t.unsqueeze(-1)
    return xa

def gaussian_nll_loss(v_pred, v_gt, log_sig, min_sigma=1e-3):
    sigma=torch.nn.functional.softplus(log_sig.squeeze(-1))+min_sigma
    err=v_pred.squeeze(-1)-v_gt
    nll=0.5*(err/sigma)**2 + torch.log(sigma) + 0.5*math.log(2*math.pi)
    return nll.mean()

def train_one_epoch(model, loader, optim, device, lambda_nll=0.1, augment_yaw=False):
    model.train()
    total=total_mse=0; n=0
    for x,v,_ in loader:
        x=x.to(device); v=v.to(device)
        if augment_yaw and random.random()<0.5:
            x=augment_random_yaw(x)
        optim.zero_grad()
        v_pred, log_sig,_ ,_ ,_=model(x)
        mse=nn.functional.mse_loss(v_pred.squeeze(-1), v)
        loss=(1-lambda_nll)*mse + lambda_nll*gaussian_nll_loss(v_pred, v, log_sig) if lambda_nll>0 else mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
        total+=loss.item()*len(x); total_mse+=mse.item()*len(x); n+=len(x)
    return total/n, total_mse/n

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total=n=0
    for x,v,_ in loader:
        x=x.to(device); v=v.to(device)
        v_pred,_,_,_,_=model(x)
        total+=nn.functional.mse_loss(v_pred.squeeze(-1), v, reduction="sum").item()
        n+=len(x)
    return total/n
