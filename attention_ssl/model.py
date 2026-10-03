"""Candidate-free completion with attention keys and support-fitted fast weights.

Training and inference use the same baseline-plus-evidence energy. The frozen
baseline remains an explicit anchor, not a substitute for an acceptance test.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from sspredrnet.energy import operator_energy
from sspredrnet.model import Reasoner


class RelationKeys(nn.Module):
    """Attend across 25 spatial relation tokens without reading a completion."""
    def __init__(self):
        super().__init__()
        axis = torch.linspace(-1, 1, 5)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y, x.square(), y.square()), -1).reshape(25, 4))
        self.input = nn.Sequential(nn.LayerNorm(100), nn.Linear(100, 64), nn.GELU())
        self.attention_norm = nn.LayerNorm(64)
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 32))

    def forward(self, prefix):
        first, second = F.relu(prefix).unbind(1)
        tokens = torch.cat((first, second, second - first), 1).transpose(1, 2)
        position = self.coordinates.to(tokens).expand(len(tokens), -1, -1)
        tokens = self.input(torch.cat((tokens, position), -1))
        normed = self.attention_norm(tokens)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        return F.normalize(self.output(tokens + attended).float(), dim=-1, eps=1e-3)


class SupportCompletion(nn.Module):
    """A support row fits a regularized relation map; queries only read that map.

Ridge penalizes an ill-conditioned contextual fit. It is not an anti-collapse
objective. The target encoder is immutable and CE negatives are known panels.
"""
    def __init__(self):
        super().__init__()
        self.keys = RelationKeys()
        self.prior = nn.Linear(64, 32)
        with torch.no_grad():
            self.prior.weight.zero_()
            self.prior.weight[:, 32:].copy_(torch.eye(32))
            self.prior.bias.zero_()
        self.gate = nn.Parameter(torch.zeros(()))
        self.ridge = 1.

    def unconditional(self, prefix):
        tokens = F.relu(prefix).transpose(2, 3).permute(0, 2, 1, 3).flatten(2)
        return self.prior(tokens).transpose(1, 2)

    def fit(self, support):
        keys = self.keys(support[:, :2])
        values = (F.relu(support[:, 2]) - self.unconditional(support[:, :2])).transpose(1, 2)
        # Solve in FP32 even under mixed precision; never explicitly invert.
        with torch.autocast(device_type=support.device.type, enabled=False):
            keys, values = keys.float(), values.float()
            gram = keys @ keys.transpose(1, 2)
            identity = torch.eye(25, device=keys.device, dtype=keys.dtype)
            coefficients = torch.linalg.solve(gram + self.ridge * identity, values)
        return keys, coefficients

    def forward(self, support, prefix):
        keys, coefficients = self.fit(support)
        queries = self.keys(prefix)
        prior = self.unconditional(prefix)
        with torch.autocast(device_type=prefix.device.type, enabled=False):
            weights = queries.float() @ keys.transpose(1, 2)
            correction = (weights @ coefficients).transpose(1, 2)
            conditional = prior.float() + self.gate.float().tanh() * correction
        return conditional, prior.float()


class AttentionCompletion(nn.Module):
    """Frozen SSPredRNet plus a learned, candidate-independent support map."""
    def __init__(self, *, evidence_weight=.2):
        super().__init__()
        if not math.isfinite(evidence_weight) or evidence_weight <= 0:
            raise ValueError('evidence_weight must be finite and positive')
        self.reasoner = Reasoner().requires_grad_(False).eval()
        self.completion = SupportCompletion()
        self.evidence_weight = evidence_weight
        self.temperature = .15
        self.alignment_weight = .1

    def load_baseline(self, state):
        self.reasoner.load_state_dict({k.removeprefix('reasoner.'): v for k, v in state.items()}, strict=True)

    def train(self, mode=True):
        super().train(mode)
        self.reasoner.eval()
        return self

    def features(self, views):
        # The immutable teacher and anchor keep the native FP32 scoring path
        # during both adaptation and validation, including under AMP.
        with torch.no_grad(), torch.autocast(device_type=views.device.type, enabled=False):
            return self.reasoner._features(views.float().flatten(0, 1))

    def completion_energy(self, support, prefix, targets):
        conditional, prior = self.completion(support, prefix)
        conditional_energy = operator_energy(conditional[:, None], targets)[:, 0]
        unconditional_energy = operator_energy(prior[:, None], targets)[:, 0]
        delta = conditional_energy - unconditional_energy
        return delta, conditional_energy

    def row_scores(self, support, prefix, targets):
        delta, conditional_energy = self.completion_energy(support, prefix, targets)
        with torch.no_grad(), torch.autocast(device_type=support.device.type, enabled=False):
            context = torch.cat((support, prefix), 1)
            matrices = torch.cat((context[:, None].expand(-1, targets.shape[1], -1, -1, -1),
                                  targets.unsqueeze(2)), 2)
            baseline = self.reasoner._errors(matrices.float())[-1]
        return baseline + self.evidence_weight * delta, conditional_energy, baseline

    def predict_features(self, features, batch):
        targets = features[:, 8:]
        first = self.row_scores(features[:, :3], features[:, 6:8], targets)[0]
        second = self.row_scores(features[:, 3:6], features[:, 6:8], targets)[0]
        return (first + second).reshape(batch, 2, 8).mean(1)

    def forward(self, views):
        if tuple(views.shape[1:]) != (2, 16, 80, 80):
            raise ValueError('ranking requires sixteen panels per component view')
        return self.predict_features(self.features(views), len(views))

    def ssl(self, views, configurations):
        """One direction only: first complete row + second prefix -> known sixth.

Hard negatives are other known sixth panels, matched by layout/component.
Mining uses immutable target-feature distances, never puzzle answer indices.
"""
        if tuple(views.shape[1:]) != (2, 6, 80, 80):
            raise ValueError('SSL requires exactly six known panels')
        if configurations.shape != (len(views),):
            raise ValueError('one layout index is required per puzzle')
        features = self.features(views)
        n, device = len(features), views.device
        ids = torch.arange(n, device=device)
        configs = configurations.repeat_interleave(2)
        with torch.no_grad():
            target = features[:, 5]
            distance = operator_energy(F.relu(target)[:, None], target)[:, 0]
            same_group = ((configs[:, None] == configs[None]) &
                          (ids[:, None].remainder(2) == ids[None].remainder(2)))
            pixels = views.flatten(0, 1)[:, 5].flatten(1).float()
            duplicate = torch.cdist(pixels, pixels, p=0).eq(0)
            distance = distance.masked_fill(~same_group | duplicate, torch.inf)
            negative_distance, negative_ids = distance.topk(min(7, n - 1), largest=False)
            selected = torch.cat((ids[:, None], negative_ids), 1)
            valid = torch.cat((torch.ones(n, 1, dtype=torch.bool, device=device),
                               torch.isfinite(negative_distance)), 1)
        scores, energy, baseline = self.row_scores(features[:, :3], features[:, 3:5], target[selected])
        logits = (-scores / self.temperature).masked_fill(~valid, -torch.inf)
        contrastive = F.cross_entropy(logits, torch.zeros(n, dtype=torch.long, device=device))
        alignment = energy[:, 0].mean()
        loss = contrastive + self.alignment_weight * alignment
        with torch.no_grad():
            baseline_logits = (-baseline / self.temperature).masked_fill(~valid, -torch.inf)
            eligible = valid[:, 1:].any(1)
            count = int(eligible.sum())
            correct = int(((logits.argmax(1) == 0) & eligible).sum())
            base_correct = int(((baseline_logits.argmax(1) == 0) & eligible).sum())
        return loss, {'contrastive': float(contrastive.detach()), 'alignment': float(alignment.detach()),
                      'eligible_component_queries': count, 'retrieval_correct': correct,
                      'baseline_retrieval_correct': base_correct,
                      'negative_count_mean': float(valid[:, 1:].float().sum(1).mean())}
