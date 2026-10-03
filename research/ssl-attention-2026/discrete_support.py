"""Matched research comparator: six executable operators instead of a fast map.

Keep attention_ssl's task, hard negatives, frozen baseline, deployed energy
difference and validation selection. Only the support completion mechanism is
replaced. Candidate features never affect support weights. This is a comparator,
not a user-selectable mode of the primary runtime.
"""
import torch
from torch import nn

from program_ssl.model import CompletionOperators, ProgramPrior, operator_energy


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
