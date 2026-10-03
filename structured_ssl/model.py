"""Compose frozen scoring anchor, perception, support PaV and SSL objective."""
import torch
from torch import nn
from attention_ssl.model import AttentionCompletion
from sspredrnet.energy import operator_energy as dense_energy
from .constants import WIDTH, STAGES, OBJECTS
from .perception import ObjectPerception, MaskedObjectPredictor, target_objects
from .pav import IterativePaV
from .energy import set_energy
from .objectives import completion_ssl


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

    def ssl(self, pack, configurations):
        return completion_ssl(self, pack, configurations)
