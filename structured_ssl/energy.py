"""Inference discrepancy and transport-aligned support feedback in FP32."""
import math
import torch
from torch.nn import functional as F
from sspredrnet.energy import operator_energy as dense_energy
from .constants import OBJECTS


def object_transport(predicted, presence, targets, valid):
    """Balanced entropic transport includes null targets for count prediction.

    Predicted and target object ordering may differ. Cost uses fixed appearance,
    explicit geometry and presence. Ten iterations run in FP32.
    """
    with torch.autocast(device_type=predicted.device.type, enabled=False):
        p, t = predicted.float(), targets.float()
        appearance = 1 - torch.einsum('nid,njkd->njik', F.normalize(p[..., :32], dim=-1, eps=1e-3),
                                      F.normalize(t[..., :32], dim=-1, eps=1e-3))
        geometry = (p[:, None, :, None, 32:] - t[:, :, None, :, 32:]).square().mean(-1)
        occupied = valid[:, :, None, :].float()
        logits = presence.float()[:, None, :, None]
        occupancy = F.softplus(logits) - logits * occupied
        cost = occupied * (appearance.clamp_min(0) + geometry) + .25 * occupancy
        log_kernel = -cost / .1
        u = torch.zeros_like(cost[..., 0])
        v = torch.zeros_like(cost[..., 0, :])
        marginal = -math.log(OBJECTS)
        for _ in range(10):
            u = marginal - torch.logsumexp(log_kernel + v[..., None, :], -1)
            v = marginal - torch.logsumexp(log_kernel + u[..., :, None], -2)
        transport = (log_kernel + u[..., :, None] + v[..., None, :]).exp()
        return cost, transport



def set_energy(predicted, presence, targets, valid):
    cost, transport = object_transport(predicted, presence, targets, valid)
    return (transport * cost).sum((-1, -2))


def prediction_energies(predictions, priors, targets):
    """Average the same dense/object discrepancy over all prediction stages."""
    conditional, unconditional = [], []
    for predicted, prior in zip(predictions, priors):
        stage_energies = []
        for dense, objects, presence in (predicted, prior):
            dense_error = dense_energy(dense[:, None], targets.dense)[:, 0]
            objects_error = set_energy(objects, presence, targets.objects, targets.valid)
            stage_energies.append(dense_error + .25 * objects_error)
        conditional.append(stage_energies[0])
        unconditional.append(stage_energies[1])
    return torch.stack(conditional).mean(0), torch.stack(unconditional).mean(0)



def support_residuals(prediction, dense_target, object_target, valid):
    """Preserve all spatial errors and align object errors by the scoring plan.

    Local and pooled dense residuals follow the deployed normalized energy.
    Object values are transported into predicted-slot order; empty targets
    contribute occupancy error, not arbitrary appearance/geometry values.
    """
    dense, objects, presence = prediction
    with torch.autocast(device_type=dense.device.type, enabled=False):
        dense, truth = dense.float(), F.relu(dense_target.float())
        local = (F.normalize(dense, dim=1, eps=1e-3) -
                 F.normalize(truth, dim=1, eps=1e-3)).transpose(1, 2)
        pooled = (F.normalize(dense.mean(-1), dim=1, eps=1e-3) -
                  F.normalize(truth.mean(-1), dim=1, eps=1e-3))
        dense_residual = torch.cat((local, pooled[:, None].expand(-1, 25, -1)), -1)
        _, transport = object_transport(objects, presence, object_target[:, None], valid[:, None])
        assignment = transport[:, 0]
        assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-8)
        occupancy = (assignment @ valid.float().unsqueeze(-1)).squeeze(-1)
        target_values = torch.cat((F.normalize(object_target[..., :32].float(), dim=-1, eps=1e-3),
                                   object_target[..., 32:].float()), -1) * valid[..., None]
        matched = assignment @ target_values
        predicted_values = torch.cat((F.normalize(objects[..., :32].float(), dim=-1, eps=1e-3),
                                      objects[..., 32:].float()), -1)
        difference = predicted_values * occupancy[..., None] - matched
        object_residual = torch.cat((difference, (presence.float().sigmoid() - occupancy)[..., None]), -1)
        return dense_residual, object_residual
