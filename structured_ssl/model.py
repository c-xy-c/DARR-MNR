"""Compose frozen scoring anchor, perception, support PaV and SSL objective."""
import torch
from torch import nn
from attention_ssl.model import AttentionCompletion
from .constants import WIDTH, STAGES, OBJECTS
from .perception import ObjectPerception, MaskedObjectPredictor, target_objects
from .pav import IterativePaV
from .energy import prediction_energies
from .representations import VisibleContext, PanelTargets
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

    def encode_context(self, pack, count):
        flat = {k: v.flatten(0, 1)[:, :count] for k, v in pack.items()}
        n = len(flat['views'])
        levels = self.perception(flat['views'].flatten(0, 1), flat['masks'].flatten(0, 1),
                                 flat['geometry'].flatten(0, 1), flat['valid'].flatten(0, 1))
        return VisibleContext(levels.reshape(n, count, STAGES, OBJECTS + 1, WIDTH), flat['valid'])

    def fixed_targets(self, pack):
        dense = self.anchor.features(pack['views'])
        flat = {k: v.flatten(0, 1) for k, v in pack.items()}
        objects = target_objects(self.anchor, flat['views'], flat['masks'], flat['boxes'],
                                 flat['geometry'], flat['valid'])
        return PanelTargets(dense, objects, flat['valid'])

    def score_row(self, context, teacher, *, support, prefix, completion, order=None):
        selected = torch.arange(len(context.levels), device=context.levels.device) if order is None else order
        observed = context.select(support, selected)
        reference = teacher.select(support, selected)
        query = context.select(prefix)
        predictions, priors, reference_error, factors = self.pav(
            observed.levels, observed.valid, reference.dense[:, 2],
            reference.objects[:, 2], reference.valid[:, 2], query.levels, query.valid)
        # Existing attention anchor uses the real support even in intervention.
        with torch.no_grad(), torch.autocast(device_type=teacher.dense.device.type, enabled=False):
            anchor, _, _ = self.anchor.row_scores(teacher.dense[:, support], teacher.dense[:, prefix],
                                                 teacher.dense[:, completion])
        conditional, unconditional = prediction_energies(predictions, priors, teacher.select(completion))
        score = anchor + self.evidence_weight * self.gate.tanh() * (conditional - unconditional)
        return score, conditional, {'anchor': anchor, 'factors': factors, 'reference_error': reference_error}

    def rank(self, context, teacher, batch_size, *, order=None):
        """Both support rows and both component views share one ranking path."""
        if context.levels.shape[1] != 8 or teacher.dense.shape[1] != 16:
            raise ValueError('ranking requires eight visible panels and sixteen fixed targets')
        scores, anchors, factors = [], [], []
        for support in (slice(0, 3), slice(3, 6)):
            observed, _, evidence = self.score_row(context, teacher, support=support,
                prefix=slice(6, 8), completion=slice(8, 16), order=order)
            scores.append(observed)
            anchors.append(evidence['anchor'])
            factors.append(evidence['factors'])
        return (sum(scores).reshape(batch_size, 2, 8).mean(1),
                {'anchor': sum(anchors).reshape(batch_size, 2, 8).mean(1), 'factors': tuple(factors)})

    def forward(self, pack):
        if tuple(pack['views'].shape[1:]) != (2, 16, 80, 80):
            raise ValueError('ranking requires sixteen panels')
        teacher = self.fixed_targets(pack)
        context = self.encode_context(pack, 8)  # Candidates never enter trainable context.
        return self.rank(context, teacher, len(pack['views']))[0]

    def ssl(self, pack, configurations):
        return completion_ssl(self, pack, configurations)
