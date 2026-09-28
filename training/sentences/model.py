"""Conformer-lite encoder + CTC head. Input SF1 [B, T, 356] at 15 fps, output log-probs at 7.5 fps."""
import torch
from torch import nn

CONFIGS = {
    "large": dict(d=384, layers=12, heads=6, ff=1536, conv=15),
    "small": dict(d=192, layers=6, heads=4, ff=512, conv=15),
}
MAX_STEPS = 1024  # positions after subsampling: 2 min 16 s of signing (the app sends at most 30 s)


class ConvModule(nn.Module):
    def __init__(self, d, k, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # mask: [B, T], True = padding
        y = nn.functional.glu(self.pw1(self.norm(x)), dim=-1)
        y = y.masked_fill(mask[..., None], 0).transpose(1, 2)
        y = nn.functional.silu(self.bn(self.dw(y))).transpose(1, 2)
        return x + self.drop(self.pw2(y))


class SelfAttention(nn.Module):
    """Multi-head self-attention with nn.MultiheadAttention's parameter names (checkpoints load
    either way) but shape-agnostic reshapes, so the ONNX export works for any length."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, mask):  # mask: [B, T], True = padding
        B, _, d = x.shape
        qkv = nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.reshape(B, -1, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        keep = ~mask[:, None, None, :]
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep,
                                                       dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(y.transpose(1, 2).reshape(B, -1, d))


class Block(nn.Module):
    def __init__(self, d, heads, ff, conv, dropout):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.norm_att = nn.LayerNorm(d)
        self.att = SelfAttention(d, heads, dropout)
        self.conv = ConvModule(d, conv, dropout)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.norm_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + 0.5 * self.drop(self.ff1(x))
        a = self.norm_att(x)
        x = x + self.drop(self.att(a, mask))
        x = self.conv(x, mask)
        x = x + 0.5 * self.drop(self.ff2(x))
        return self.norm_out(x)


class SignCTC(nn.Module):
    def __init__(self, n_classes, d, layers, heads, ff, conv, dropout=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(356, d), nn.LayerNorm(d), nn.SiLU())
        self.sub = nn.Conv1d(d, d, 3, stride=2, padding=1)  # 15 fps -> 7.5 fps
        self.pos = nn.Parameter(torch.zeros(1, MAX_STEPS, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, ff, conv, dropout) for _ in range(layers))
        self.head = nn.Linear(d, n_classes)

    def forward(self, x, lengths):
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        h = self.inp(x) * valid[..., None]
        h = self.sub(h.transpose(1, 2)).transpose(1, 2)
        out_len = (lengths + 1) // 2
        mask = torch.arange(h.shape[1], device=x.device)[None] >= out_len[:, None]
        h = h + self.pos[:, : h.shape[1]]
        for b in self.blocks:
            h = b(h, mask)
        return self.head(h).log_softmax(-1), out_len


def build(name, n_classes, dropout=0.1):
    return SignCTC(n_classes, **CONFIGS[name], dropout=dropout)
