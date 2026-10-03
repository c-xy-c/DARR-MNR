"""FP32 spatial and pooled feature discrepancy shared by completion models."""
import torch
from torch.nn import functional as F

def operator_energy(predictions, targets):
    """N x K x J energy from spatial and pooled prediction discrepancy."""
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        predictions, targets = predictions.float(), F.relu(targets.float())
        shared = targets.ndim == 3
        target_channels = 1 if shared else 2
        p = F.normalize(predictions, dim=2, eps=1e-3)
        t = F.normalize(targets, dim=target_channels, eps=1e-3)
        target_norm = t.square().sum(target_channels).mean(-1)
        target_norm = target_norm[None, None] if shared else target_norm[:, None]
        dense = p.square().sum(2).mean(-1)[:, :, None] + target_norm
        equation = 'nkcl,mcl->nkm' if shared else 'nkcl,njcl->nkj'
        dense = dense - 2 * torch.einsum(equation, p, t) / p.shape[-1]
        p = F.normalize(predictions.mean(-1), dim=2, eps=1e-3)
        t = F.normalize(targets.mean(-1), dim=target_channels, eps=1e-3)
        target_norm = t.square().sum(target_channels)
        target_norm = target_norm[None, None] if shared else target_norm[:, None]
        pooled = p.square().sum(2)[:, :, None] + target_norm
        equation = 'nkc,mc->nkm' if shared else 'nkc,njc->nkj'
        pooled = pooled - 2 * torch.einsum(equation, p, t)
        return (.5 * (dense + pooled)).clamp_min(0)
