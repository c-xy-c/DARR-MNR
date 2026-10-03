"""Trainable object perception and three-stage support-compiled PaV.

The immutable attention baseline is the scoring anchor and target encoder.
Every dynamic parameter and refinement signal comes from complete support rows;
candidates enter only the final discrepancy calculation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from attention_ssl.model import AttentionCompletion
from program_ssl.model import operator_energy
from .data import OBJECTS


WIDTH, STAGES, RANK = 96, 3, 8


class SelfBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(WIDTH, 6, WIDTH * 4, dropout=0,
                                                batch_first=True, norm_first=True, activation='gelu')

    def forward(self, x, padding=None):
        return self.block(x, src_key_padding_mask=padding)


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_norm, self.memory_norm = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 6, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, WIDTH * 4),
                                 nn.GELU(), nn.Linear(WIDTH * 4, WIDTH))

    def forward(self, queries, memory, padding=None, mask=None):
        q, m = self.query_norm(queries), self.memory_norm(memory)
        update, _ = self.attention(q, m, m, key_padding_mask=padding, attn_mask=mask, need_weights=False)
        out = queries + update
        return out + self.ffn(out)


class ObjectPerception(nn.Module):
    """Raw 8px patches -> three levels -> masked mean/cross-attention pooling."""
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(1, WIDTH, 8, stride=8)
        axis = torch.linspace(-1, 1, 10)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y, x.square(), y.square()), -1).reshape(100, 4))
        self.position = nn.Linear(4, WIDTH)
        self.geometry = nn.Linear(6, WIDTH)
        self.layers = nn.ModuleList([SelfBlock() for _ in range(STAGES)])
        self.poolers = nn.ModuleList([CrossBlock() for _ in range(STAGES)])
        self.activation_checkpointing = True

    def forward(self, images, masks, geometry, valid):
        tokens = self.patches(images[:, None]).flatten(2).transpose(1, 2)
        tokens = tokens + self.position(self.coordinates.to(tokens))
        weights = masks.to(tokens) / masks.to(tokens).sum(-1, keepdim=True).clamp_min(1e-6)
        # Inactive objects use one harmless key, then their outputs are erased.
        first_key = torch.arange(100, device=masks.device).eq(0)
        outside = (masks <= 0) & (valid[..., None] | ~first_key)
        outside = outside.repeat_interleave(6, dim=0)
        levels = []
        for layer, pooler in zip(self.layers, self.poolers):
            recompute = self.training and torch.is_grad_enabled() and self.activation_checkpointing
            tokens = checkpoint(layer, tokens, use_reentrant=False) if recompute else layer(tokens)
            mean = weights @ tokens
            query = mean + self.geometry(geometry.to(tokens))
            objects = (checkpoint(pooler, query, tokens, mask=outside, use_reentrant=False)
                       if recompute else pooler(query, tokens, mask=outside))
            objects = objects * valid[..., None]
            scene = tokens.mean(1, keepdim=True)
            levels.append(torch.cat((objects, scene), 1))
        return torch.stack(levels, 1)


class MemoryCompiler(nn.Module):
    """SHINE-inspired depth/token attention produces coupled P/G/V factors.

    Each depth has 48 memory tokens: 3 roles x 2 factors x rank8 tokens.
    This contains exactly enough elements for three rank8 96x96 updates.
    """
    def __init__(self):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(STAGES, 48, WIDTH) * .02)
        self.column = nn.Parameter(torch.randn(3, WIDTH) * .02)
        self.extract = nn.ModuleList([CrossBlock() for _ in range(STAGES)])
        self.axial = nn.ModuleList([SelfBlock() for _ in range(4)])
        self.output = nn.LayerNorm(WIDTH)

    def forward(self, support, valid):
        # support: N x three panels x three depths x eleven object/scene tokens.
        n = len(support)
        padding = torch.cat((~valid, torch.zeros(n, 3, 1, device=valid.device, dtype=torch.bool)), -1).flatten(1)
        memories = []
        for depth, block in enumerate(self.extract):
            observed = (support[:, :, depth] + self.column[None, :, None]).flatten(1, 2)
            memories.append(block(self.memory[depth][None].expand(n, -1, -1), observed, padding))
        z = torch.stack(memories, 1)
        for index, block in enumerate(self.axial):
            if index % 2 == 0:
                z = block(z.transpose(1, 2).reshape(n * 48, STAGES, WIDTH)).reshape(n, 48, STAGES, WIDTH).transpose(1, 2)
            else:
                z = block(z.reshape(n * STAGES, 48, WIDTH)).reshape(n, STAGES, 48, WIDTH)
        factors = self.output(z).reshape(n, STAGES, 3, 2, RANK, WIDTH)
        return F.normalize(factors.float(), dim=-1, eps=1e-3)


def dynamic(x, factors):
    with torch.autocast(device_type=x.device.type, enabled=False):
        a, b = factors.float().unbind(1)
        return ((x.float() @ a.transpose(1, 2)) @ b) / math.sqrt(RANK)


def dense_energy(predictions, targets):
    # The inherited helper disables CUDA autocast. Also guard the actual
    # device, so a Metal/CPU autocast context cannot change the fixed energy.
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        return operator_energy(predictions.float(), targets.float())


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


def target_objects(teacher, images, masks, boxes, geometry, valid):
    """Encode each bbox on a white panel before pooling its region features.

    Pixels, coordinates and scale inside the box stay unchanged. Context outside
    it cannot affect the appearance target. Overlapping boxes can still include
    another object's pixels; exterior contours are not semantic segmentation.
    Chunking bounds teacher activations without changing targets or gradients.
    """
    with torch.no_grad(), torch.autocast(device_type=images.device.type, enabled=False):
        n, panels = images.shape[:2]
        pooled_masks = F.avg_pool2d(masks.float().reshape(n * panels * OBJECTS, 1, 10, 10), 2)
        weights = pooled_masks.reshape(-1, 25)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        appearance = torch.zeros(n * panels * OBJECTS, 32, device=images.device)
        active = valid.flatten().nonzero().flatten()
        flat_images, flat_boxes = images.float().flatten(0, 1), boxes.reshape(-1, 4)
        grid = torch.arange(80, device=images.device)
        for start in range(0, len(active), 128):
            ids = active[start:start + 128]
            x0, y0, x1, y1 = flat_boxes[ids].unbind(-1)
            inside = ((grid[None, None, :] >= x0[:, None, None]) &
                      (grid[None, None, :] < x1[:, None, None]) &
                      (grid[None, :, None] >= y0[:, None, None]) &
                      (grid[None, :, None] < y1[:, None, None]))
            isolated = flat_images[ids // OBJECTS].masked_fill(~inside, 1.)
            features = teacher.features(isolated[:, None, None])[:, 0]
            pooled = (weights[ids, None] @ F.relu(features).transpose(1, 2)).squeeze(1)
            appearance.index_copy_(0, ids, pooled)
        appearance = appearance.reshape(n, panels, OBJECTS, 32)
        return torch.cat((appearance, geometry.float()), -1) * valid[..., None]


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


class SupportFeedback(nn.Module):
    """Query states attend to support states with structured errors as values."""
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, WIDTH, bias=False)
        self.objects = nn.Linear(39, WIDTH, bias=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = nn.MultiheadAttention(WIDTH, 6, batch_first=True, bias=False)

    def forward(self, state, reference, residuals):
        values = torch.cat((self.dense(residuals[0]), self.objects(residuals[1])), 1)
        update, _ = self.attention(self.norm(state), self.norm(reference), values, need_weights=False)
        return update


class IterativePaV(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiler = MemoryCompiler()
        self.queries = nn.Parameter(torch.randn(25 + OBJECTS, WIDTH) * .02)
        self.column = nn.Parameter(torch.randn(2, WIDTH) * .02)
        self.initialize = CrossBlock()
        self.updates = nn.ModuleList([SelfBlock() for _ in range(STAGES)])
        self.dense = nn.ModuleList([nn.Linear(WIDTH, 32) for _ in range(STAGES)])
        self.objects = nn.ModuleList([nn.Linear(WIDTH, 38) for _ in range(STAGES)])
        self.presence = nn.ModuleList([nn.Linear(WIDTH, 1) for _ in range(STAGES)])
        self.feedback = nn.ModuleList([SupportFeedback() for _ in range(STAGES - 1)])

    def initial(self, prefix, valid):
        n = len(prefix)
        memory = (prefix[:, :, -1] + self.column[None, :, None]).flatten(1, 2)
        padding = torch.cat((~valid, torch.zeros(n, 2, 1, device=valid.device, dtype=torch.bool)), -1).flatten(1)
        return self.initialize(self.queries[None].expand(n, -1, -1), memory, padding)

    def decode(self, state, stage):
        return (self.dense[stage](state[:, :25]).transpose(1, 2),
                self.objects[stage](state[:, 25:]), self.presence[stage](state[:, 25:]).squeeze(-1))

    def forward(self, support, support_valid, support_target, support_objects, support_object_valid,
                prefix, prefix_valid, *, factors=None):
        if factors is None:
            factors = self.compiler(support, support_valid)
        query = self.initial(prefix, prefix_valid)
        reference = self.initial(support[:, :2], support_valid[:, :2])
        prior = query.clone()
        outputs, priors, reference_errors = [], [], []
        for stage in range(STAGES):
            weights = factors[:, stage]
            query, reference, prior = [self.updates[stage](x) for x in (query, reference, prior)]
            query = query + .1 * dynamic(query, weights[:, 0])
            reference = reference + .1 * dynamic(reference, weights[:, 0])
            verified_query = query + .1 * dynamic(query, weights[:, 1]).sigmoid() * dynamic(query, weights[:, 2])
            verified_reference = reference + .1 * dynamic(reference, weights[:, 1]).sigmoid() * dynamic(reference, weights[:, 2])
            prediction = self.decode(verified_query, stage)
            support_prediction = self.decode(verified_reference, stage)
            outputs.append(prediction)
            priors.append(self.decode(prior, stage))
            residuals = support_residuals(support_prediction, support_target, support_objects, support_object_valid)
            reference_errors.append(residuals[0].square().mean((1, 2)) + .25 * residuals[1].square().mean((1, 2)))
            if stage < STAGES - 1:
                # Compile once; support reconstruction discrepancy refines the
                # next query stage without candidate-dependent feedback.
                for name, state in (('query', query), ('reference', reference)):
                    measured = self.feedback[stage](state, verified_reference, residuals)
                    gate = dynamic(state, weights[:, 1]).sigmoid()
                    refined = state - .1 * gate * dynamic(measured, weights[:, 2])
                    if name == 'query':
                        query = refined
                    else:
                        reference = refined
        return outputs, priors, torch.stack(reference_errors, -1), factors


class MaskedObjectPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, WIDTH) * .02)
        self.panel = nn.Parameter(torch.randn(5, WIDTH) * .02)
        self.center = nn.Linear(2, WIDTH)
        self.layers = nn.ModuleList([CrossBlock(), CrossBlock()])
        self.output = nn.Linear(WIDTH, 32)

    def forward(self, levels, valid, centers, panel_indices):
        n = len(levels)
        queries = self.query + self.panel[panel_indices] + self.center(centers)
        memory = (levels[:, :, -1] + self.panel[None, :, None]).flatten(1, 2)
        padding = torch.cat((~valid, torch.zeros(n, 5, 1, device=valid.device, dtype=torch.bool)), -1).flatten(1)
        for layer in self.layers:
            queries = layer(queries, memory, padding)
        return self.output(queries)


class StructuredCompletion(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = AttentionCompletion().requires_grad_(False).eval()
        self.perception = ObjectPerception()
        self.pav = IterativePaV()
        self.masked = MaskedObjectPredictor()
        self.gate = nn.Parameter(torch.zeros(()))
        self.evidence_weight = .2
        self.temperature = .15

    def load_anchor(self, state):
        self.anchor.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def encode(self, pack, count):
        flat = {k: v.flatten(0, 1)[:, :count] for k, v in pack.items()}
        n = len(flat['views'])
        levels = self.perception(flat['views'].flatten(0, 1), flat['masks'].flatten(0, 1),
                                 flat['geometry'].flatten(0, 1), flat['valid'].flatten(0, 1))
        return levels.reshape(n, count, STAGES, OBJECTS + 1, WIDTH), flat

    def targets(self, pack):
        flat = {k: v.flatten(0, 1) for k, v in pack.items()}
        return target_objects(self.anchor, flat['views'], flat['masks'], flat['boxes'],
                              flat['geometry'], flat['valid']), flat['valid']

    def row(self, levels, visible, features, objects, valid, support, prefix, targets, *, order=None):
        selected = torch.arange(len(levels), device=levels.device) if order is None else order
        predictions, priors, reference_error, factors = self.pav(
            levels[selected, support], visible['valid'][selected, support], features[selected, support][:, 2],
            objects[selected, support][:, 2], valid[selected, support][:, 2],
            levels[:, prefix], visible['valid'][:, prefix])
        # Existing attention anchor uses the real support even in intervention.
        with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
            anchor, _, _ = self.anchor.row_scores(features[:, support], features[:, prefix], features[:, targets])
        conditional, unconditional = [], []
        for predicted, prior in zip(predictions, priors):
            stage_energies = []
            for output in (predicted, prior):
                dense, representation, presence = output
                dense_error = dense_energy(dense[:, None], features[:, targets])[:, 0]
                objects_error = set_energy(representation, presence, objects[:, targets], valid[:, targets])
                stage_energies.append(dense_error + .25 * objects_error)
            conditional.append(stage_energies[0])
            unconditional.append(stage_energies[1])
        conditional, unconditional = torch.stack(conditional).mean(0), torch.stack(unconditional).mean(0)
        score = anchor + self.evidence_weight * self.gate.tanh() * (conditional - unconditional)
        return score, conditional, {'anchor': anchor, 'factors': factors, 'reference_error': reference_error}

    def forward(self, pack):
        if tuple(pack['views'].shape[1:]) != (2, 16, 80, 80):
            raise ValueError('ranking requires sixteen panels')
        features = self.anchor.features(pack['views'])
        levels, visible = self.encode(pack, 8)  # Candidates never enter perception/context compilation.
        objects, valid = self.targets(pack)
        scores = [self.row(levels, visible, features, objects, valid, support, slice(6, 8), slice(8, 16))[0]
                  for support in (slice(0, 3), slice(3, 6))]
        return sum(scores).reshape(len(pack['views']), 2, 8).mean(1)

    def masked_loss(self, pack, target_objects_fixed):
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
        levels = self.perception(masked_views.flatten(0, 1), masked_masks.flatten(0, 1),
                                 masked_geometry.flatten(0, 1), masked_valid.flatten(0, 1))
        levels = levels.reshape(n, 5, STAGES, OBJECTS + 1, WIDTH)
        predicted = self.masked(levels, masked_valid, centers, panels[:, None])
        truth = target_objects_fixed[:, :5][rows, panels, selected, :32].unsqueeze(1)
        discrepancy = (F.normalize(predicted.float(), dim=-1, eps=1e-3) -
                       F.normalize(truth.float(), dim=-1, eps=1e-3)).square().sum(-1)
        return (discrepancy * active[:, None]).sum() / active.sum().clamp_min(1)

    def ssl(self, pack, configurations):
        if tuple(pack['views'].shape[1:]) != (2, 6, 80, 80):
            raise ValueError('SSL accepts exactly the six known panels')
        features = self.anchor.features(pack['views'])
        objects, valid = self.targets(pack)
        levels, visible = self.encode(pack, 5)  # The true sixth never enters a predictor.
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
        scores, energy, details = self.row(levels, visible, columns, target_set, target_valid,
                                           slice(0, 3), slice(3, 5), slice(5, None))
        logits = (-scores / self.temperature).masked_fill(~legal, -torch.inf)
        retrieval = F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=device))
        completion = energy[:, 0].mean()
        masked = self.masked_loss(pack, objects)
        loss = retrieval + .1 * completion + .1 * masked
        return loss, {'retrieval': float(retrieval.detach()), 'completion': float(completion.detach()),
                      'masked_object': float(masked.detach()), 'reference_error': float(details['reference_error'].detach().mean()),
                      'eligible_queries': int(legal[:, 1:].any(-1).sum()),
                      'retrieval_correct': int(((logits.argmax(1) == 0) & legal[:, 1:].any(-1)).sum()),
                      'proposal_overflow_panels': int(pack['overflow'].sum())}
