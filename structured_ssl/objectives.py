"""Known-cell retrieval and complete-object masking for V3 adaptation."""
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
