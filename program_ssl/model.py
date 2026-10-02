"""Executable completion operators with support-calibrated verification.

The auxiliary task cannot read support features outside its compiled operator
distribution. The published three-block PaV remains the main scoring path.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from sspredrnet.model import Reasoner


class CompletionOperators(nn.Module):
    """Distinct executable affine priors with learned rank-four corrections.

    Coefficients are learned, not labels or guarantees of semantic rules.
    """
    def __init__(self, channels=32, rank=4):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([
            [0., 1.], [1., 0.], [-1., 2.], [.5, .5], [1., 1.], [2., -1.]]))
        left = torch.empty(6, rank, 2 * channels)
        nn.init.kaiming_uniform_(left, a=math.sqrt(5))
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(torch.zeros(6, channels, rank))

    def forward(self, prefix):
        first, second = F.relu(prefix).unbind(1)
        prior = (self.coefficients[None, :, :1, None] * first[:, None] +
                 self.coefficients[None, :, 1:, None] * second[:, None])
        weights = torch.einsum('kcr,kri->kci', self.right, self.left)
        correction = torch.einsum('kci,nil->nkcl', weights, torch.cat((first, second), 1))
        return prior + .5 * torch.tanh(correction)


class ProgramPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(2400), nn.Linear(2400, 64),
                                 nn.LeakyReLU(.1), nn.Linear(64, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, support):
        first, second, third = support.unbind(1)
        return self.net(torch.stack((first, second - first, third - second), 1).flatten(1)).float()


def operator_energy(predictions, targets):
    """N x K x J energy from spatial and pooled prediction discrepancy."""
    with torch.autocast(device_type='cuda', enabled=False):
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


class ComponentProgram(nn.Module):
    """Frozen baseline plus candidate-independent completion programs."""
    def __init__(self, *, use_program=True, program_weight=.1,
                 support_temperature=.15, verify_temperature=.15,
                 static_program=False, margin=.7, verification_mode='posterior'):
        super().__init__()
        self.reasoner = Reasoner()
        self.use_program, self.static_program = use_program, static_program
        self.program_weight = program_weight
        self.support_temperature, self.verify_temperature = support_temperature, verify_temperature
        self.margin = margin
        if verification_mode not in ('posterior', 'support_evidence_ratio'):
            raise ValueError('unknown verification energy')
        self.verification_mode = verification_mode
        if use_program:
            self.operators = CompletionOperators()
            if not static_program:
                self.prior = ProgramPrior()
        for name, parameter in self.reasoner.named_parameters():
            if use_program or name.startswith(('res', 'channel_reducer')):
                parameter.requires_grad_(False)

    def load_baseline(self, state):
        self.reasoner.load_state_dict({k.removeprefix('reasoner.'): v
                                      for k, v in state.items()}, strict=True)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        if self.use_program:
            self.reasoner.eval()
        else:
            for index in range(4):
                getattr(self.reasoner, f'res{index}').eval()
            self.reasoner.channel_reducer.eval()
        return self

    def features(self, views):
        with torch.no_grad():
            return self.reasoner._features(views.flatten(0, 1))

    def compile(self, support, *, intervention=None):
        if self.static_program or intervention == 'uniform_program':
            return torch.full((len(support), 6), 1 / 6, device=support.device)
        predicted_support = self.operators(support[:, :2])
        evidence = operator_energy(predicted_support, support[:, 2:3])[:, :, 0]
        # A bounded prior cannot drown out measured support disagreement.
        # This bounds a computational role, not a distribution regularizer.
        prior = .5 * self.prior(support).tanh()
        return (prior - evidence / self.support_temperature).softmax(-1)

    def verify(self, prefix, targets, alpha):
        energies = operator_energy(self.operators(prefix), targets)
        logits = alpha.clamp_min(1e-12).log()[:, :, None] - energies / self.verify_temperature
        return -self.verify_temperature * torch.logsumexp(logits, dim=1)

    def verification_energy(self, prefix, targets, alpha):
        posterior = self.verify(prefix, targets, alpha)
        if self.verification_mode == 'posterior':
            return posterior
        uniform = torch.full_like(alpha, 1 / alpha.shape[-1])
        return posterior - self.verify(prefix, targets, uniform)

    def training_errors(self, features, program_error=None):
        targets = torch.cat((features[:, 5:6], features[:, 8:]), 1)
        n, choices = targets.shape[:2]
        context = features[:, :5].unsqueeze(1).expand(-1, choices, -1, -1, -1)
        matrices = torch.cat((context, targets.unsqueeze(2)), 2)
        errors = self.reasoner._errors(matrices)
        if program_error is not None:
            errors = [e + self.program_weight * program_error for e in errors]
        return errors

    def predict_features(self, features, batch, *, intervention=None):
        errors = []
        for support_indices in ([0, 1, 2], [3, 4, 5]):
            context = torch.cat((features[:, support_indices], features[:, 6:8]), 1)
            matrices = torch.cat((context[:, None].expand(-1, 8, -1, -1, -1),
                                  features[:, 8:].unsqueeze(2)), 2)
            errors.append(self.reasoner._errors(matrices)[-1])
        base = (errors[0] + errors[1]).reshape(batch, 2, 8).mean(1)
        if not self.use_program or intervention == 'no_program':
            return base
        first = self.compile(features[:, :3], intervention=intervention)
        second = self.compile(features[:, 3:6], intervention=intervention)
        joint_log = .5 * (first.clamp_min(1e-12).log() + second.clamp_min(1e-12).log())
        alpha = joint_log.softmax(-1)
        if intervention == 'swapped_program':
            alpha = alpha.roll(2, 0)
        program = self.verification_energy(features[:, 6:8], features[:, 8:], alpha)
        return base + self.program_weight * program.reshape(batch, 2, 8).mean(1)

    def ssl(self, views, configurations):
        """Only six known panels can physically enter the auxiliary task."""
        if tuple(views.shape[1:]) != (2, 6, 80, 80):
            raise ValueError('SSL requires exactly six known panels per component view')
        features = self.features(views)
        alpha = self.compile(features[:, :3])
        targets = features[:, 5]
        energies = self.verify(features[:, 3:5], targets, alpha)
        n = len(features)
        ids = torch.arange(n, device=views.device)
        configs = configurations.repeat_interleave(2)
        mask = (ids[:, None] % 2 != ids[None, :] % 2) | (configs[:, None] != configs[None, :])
        images = views.flatten(0, 1)[:, 5].flatten(1)
        duplicates = torch.cdist(images.float(), images.float(), p=0).eq(0)
        eye = torch.eye(n, device=views.device, dtype=torch.bool)
        logits = (-energies / self.verify_temperature).masked_fill(mask | (duplicates & ~eye), -torch.inf)
        return energies.diagonal().mean() + F.cross_entropy(logits, ids), alpha

    def forward(self, views, *, intervention=None):
        if tuple(views.shape[1:]) != (2, 16, 80, 80):
            raise ValueError('ranking requires sixteen panels per component view')
        features = self.features(views)
        if not self.training:
            return self.predict_features(features, len(views), intervention=intervention)
        alpha, program_error = None, None
        if self.use_program:
            alpha = self.compile(features[:, :3])
            targets = torch.cat((features[:, 5:6], features[:, 8:]), 1)
            program_error = self.verify(features[:, 3:5], targets, alpha)
        errors = self.training_errors(features, program_error)
        flat = views.flatten(0, 1)
        identical = (flat[:, 8:] == flat[:, 5:6]).flatten(2).all(2)
        errors = [torch.cat((e[:, :1], e[:, 1:].masked_fill(identical, self.margin)), 1)
                  for e in errors]
        return {'errors': errors, 'alpha': alpha}
