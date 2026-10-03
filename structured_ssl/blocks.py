"""Shared attention blocks; no task or checkpoint policy."""
from torch import nn
from .constants import WIDTH

class SelfBlock(nn.Module):
    def __init__(self, *, norm_first=True):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(WIDTH, 6, WIDTH * 4, dropout=0,
                                                batch_first=True, norm_first=norm_first, activation='gelu')

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
