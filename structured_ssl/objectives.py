"""Known-cell retrieval, object masking, transport energy and residuals."""
import math
import torch
from torch.nn import functional as F
from sspredrnet.energy import operator_energy as dense_energy
from .constants import WIDTH, STAGES, OBJECTS

def pixel_duplicates(images):
    """Exact known-target equality without a quadratic pixel-distance tensor.

    Byte keys compare their complete contents, so hash collisions cannot create
    false duplicates. Canonicalize signed zero to match numerical equality.
    """
    pixels = images.detach().float().cpu()
    if not torch.isfinite(pixels).all():
        raise ValueError('known-target pixels must be finite')
    array = pixels.numpy().copy()
    array[array == 0] = 0
    groups = {}
    labels = [groups.setdefault(row.tobytes(), len(groups)) for row in array]
    labels = torch.tensor(labels, device=images.device)
    return labels[:, None].eq(labels[None])


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


def masked_object_loss(model, pack, target_objects_fixed):
    flat = {k: v.flatten(0, 1)[:, :5] for k, v in pack.items()}
    n = len(flat['views'])
    # Tiny discrete draws use the checkpointed CPU RNG. This also avoids
    # device-specific multinomial behavior; metadata never changes in place.
    valid_cpu = flat['valid'].detach().cpu()
    available_panels = valid_cpu.any(-1)
    active_cpu = available_panels.any(-1)
    panel_probabilities = available_panels.float() + (~active_cpu)[:, None].float()
    panels_cpu = torch.multinomial(panel_probabilities, 1).squeeze(-1)
    object_probabilities = valid_cpu[torch.arange(n), panels_cpu].float() + (~active_cpu)[:, None].float()
    selected_cpu = torch.multinomial(object_probabilities, 1).squeeze(-1)
    device = flat['views'].device
    panels, selected, active = panels_cpu.to(device), selected_cpu.to(device), active_cpu.to(device)
    rows = torch.arange(n, device=device)
    centers = flat['geometry'][rows, panels, selected, :2].clone().unsqueeze(1)
    boxes = flat['boxes'][rows, panels, selected]
    grid = torch.arange(80, device=selected.device)
    erase = ((grid[None, None, :] >= boxes[..., 0, None, None]) &
             (grid[None, None, :] < boxes[..., 2, None, None]) &
             (grid[None, :, None] >= boxes[..., 1, None, None]) &
             (grid[None, :, None] < boxes[..., 3, None, None]))
    # One object in one panel per component query; four complete panels
    # remain even for center_single, where each panel has only one object.
    panel_mask = F.one_hot(panels, 5).bool()
    erase = panel_mask[..., None, None] & erase[:, None]
    masked_views = flat['views'].masked_fill(erase, 1.)
    # Removing clean mask/geometry prevents the masked head reading shape,
    # area or colour through proposal metadata. Only the center stays in q.
    selected_slot = panel_mask[..., None] & F.one_hot(selected, OBJECTS).bool()[:, None] & active[:, None, None]
    masked_masks = flat['masks'].masked_fill(selected_slot[..., None], 0)
    masked_geometry = flat['geometry'].masked_fill(selected_slot[..., None], 0)
    masked_valid = flat['valid'] & ~selected_slot
    levels = model.perception(masked_views.flatten(0, 1), masked_masks.flatten(0, 1),
                             masked_geometry.flatten(0, 1), masked_valid.flatten(0, 1))
    levels = levels.reshape(n, 5, STAGES, OBJECTS + 1, WIDTH)
    predicted = model.masked(levels, masked_valid, centers, panels[:, None])
    truth = target_objects_fixed[:, :5][rows, panels, selected, :32].unsqueeze(1)
    discrepancy = (F.normalize(predicted.float(), dim=-1, eps=1e-3) -
                   F.normalize(truth.float(), dim=-1, eps=1e-3)).square().sum(-1)
    return (discrepancy * active[:, None]).sum() / active.sum().clamp_min(1)


def completion_ssl(model, pack, configurations):
    if tuple(pack['views'].shape[1:]) != (2, 6, 80, 80):
        raise ValueError('SSL accepts exactly the six known panels')
    features = model.anchor.features(pack['views'])
    objects, valid = model.targets(pack)
    levels, visible = model.encode(pack, 5)  # The true sixth never enters a predictor.
    n, device = len(features), features.device
    ids = torch.arange(n, device=device)
    configs = configurations.repeat_interleave(2)
    with torch.no_grad():
        target = features[:, 5]
        distance = dense_energy(F.relu(target)[:, None], target)[:, 0]
        group = ((configs[:, None] == configs[None]) &
                 (ids[:, None].remainder(2) == ids[None].remainder(2)))
        pixels = pack['views'].flatten(0, 1)[:, 5].flatten(1).float()
        duplicate = pixel_duplicates(pixels)
        distance.masked_fill_(~group | duplicate, torch.inf)
        negative_distance, negative_ids = distance.topk(min(7, n - 1), largest=False)
        selected = torch.cat((ids[:, None], negative_ids), 1)
        legal = torch.cat((torch.ones(n, 1, dtype=torch.bool, device=device), torch.isfinite(negative_distance)), 1)
    # Append known-target negatives as scoring targets only. No candidates
    # or labels from the RAVEN answer set are read by this adaptation.
    columns = torch.cat((features[:, :5], target[selected]), 1)
    target_set = torch.cat((objects[:, :5], objects[:, 5][selected]), 1)
    target_valid = torch.cat((valid[:, :5], valid[:, 5][selected]), 1)
    scores, energy, details = model.row(levels, visible, columns, target_set, target_valid,
                                       slice(0, 3), slice(3, 5), slice(5, None))
    logits = (-scores / model.temperature).masked_fill(~legal, -torch.inf)
    retrieval = F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=device))
    completion = energy[:, 0].mean()
    masked = masked_object_loss(model, pack, objects)
    loss = retrieval + .1 * completion + .1 * masked
    return loss, {'retrieval': float(retrieval.detach()), 'completion': float(completion.detach()),
                  'masked_object': float(masked.detach()), 'reference_error': float(details['reference_error'].detach().mean()),
                  'eligible_queries': int(legal[:, 1:].any(-1).sum()),
                  'retrieval_correct': int(((logits.argmax(1) == 0) & legal[:, 1:].any(-1)).sum()),
                  'proposal_overflow_panels': int(pack['overflow'].sum())}
