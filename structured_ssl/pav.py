"""Support memory compiles P/G/V factors; residual feedback refines queries."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import SelfBlock, CrossBlock
from .constants import WIDTH, STAGES, RANK, OBJECTS
from .objectives import support_residuals

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
        # SHINE adds identity after extraction; its released M2P uses post-norm.
        self.layer_identity = nn.Parameter(torch.zeros(1, STAGES, 1, WIDTH))
        self.token_identity = nn.Parameter(torch.zeros(1, 1, 48, WIDTH))
        self.axial = nn.ModuleList([SelfBlock(norm_first=False) for _ in range(4)])
        self.output = nn.LayerNorm(WIDTH)

    def forward(self, support, valid):
        # support: N x three panels x three depths x eleven object/scene tokens.
        n = len(support)
        padding = torch.cat((~valid, torch.zeros(n, 3, 1, device=valid.device, dtype=torch.bool)), -1).flatten(1)
        memories = []
        for depth, block in enumerate(self.extract):
            observed = (support[:, :, depth] + self.column[None, :, None]).flatten(1, 2)
            memories.append(block(self.memory[depth][None].expand(n, -1, -1), observed, padding))
        z = torch.stack(memories, 1) + self.layer_identity + self.token_identity
        for index, block in enumerate(self.axial):
            if index % 2 == 0:
                z = block(z.transpose(1, 2).reshape(n * 48, STAGES, WIDTH)).reshape(n, 48, STAGES, WIDTH).transpose(1, 2)
            else:
                z = block(z.reshape(n * STAGES, 48, WIDTH)).reshape(n, STAGES, 48, WIDTH)
        return pack_factors(self.output(z))


def pack_factors(memory):
    """SHINE rl: A is width x rank, B is rank x width; executor reads A.T.

    A complete memory token is not a complete column of A. Rank projections can
    differ even when memory tokens initially look alike. Vector normalization
    is our FP32 stability adaptation, not SHINE's original scale.
    """
    blocks = memory.reshape(len(memory), STAGES, 3, 2, RANK * WIDTH)
    left = blocks[:, :, :, 0].reshape(-1, STAGES, 3, WIDTH, RANK).transpose(-1, -2)
    right = blocks[:, :, :, 1].reshape(-1, STAGES, 3, RANK, WIDTH)
    return F.normalize(torch.stack((left, right), 3).float(), dim=-1, eps=1e-3)


def dynamic(x, factors):
    with torch.autocast(device_type=x.device.type, enabled=False):
        a, b = factors.float().unbind(1)
        return ((x.float() @ a.transpose(1, 2)) @ b) / math.sqrt(RANK)


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
                prefix, prefix_valid):
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
