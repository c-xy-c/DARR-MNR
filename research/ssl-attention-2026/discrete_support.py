"""Matched research comparator: six executable operators instead of a fast map.

Keep attention_ssl's task, hard negatives, frozen baseline, deployed energy
difference and validation selection. Only the support completion mechanism is
replaced. Candidate features never affect support weights. This is a comparator,
not a user-selectable mode of the primary runtime.
"""
import math
import torch
from torch import nn

from torch.nn import functional as F
from sspredrnet.energy import operator_energy


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



class DiscreteSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.operators = CompletionOperators()
        self.prior = ProgramPrior()
        self.gate = nn.Parameter(torch.zeros(()))
        self.ridge = None

    def compile(self, support):
        predictions = self.operators(support[:, :2])
        errors = operator_energy(predictions, support[:, 2:3])[:, :, 0]
        weights = (.5 * self.prior(support).tanh() - errors / .15).softmax(-1)
        # Interpolate in log weights: zero gate is exactly the uniform reference.
        return (self.gate.float().tanh() * weights.clamp_min(1e-12).log()).softmax(-1)

    @staticmethod
    def marginal(energies, weights):
        return -.15 * torch.logsumexp(weights.clamp_min(1e-12).log()[:, :, None] - energies / .15, dim=1)

    def energies(self, support, prefix, targets):
        weights = self.compile(support)
        energies = operator_energy(self.operators(prefix), targets)
        conditional = self.marginal(energies, weights)
        uniform = self.marginal(energies, torch.full_like(weights, 1 / 6))
        return conditional - uniform, conditional
