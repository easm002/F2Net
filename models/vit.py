"""Long-range sub-branch: a 6-layer ViT-tiny (Sec. 3.3 / 4.1 of the paper)
modeling long-range dependencies over the downsampled low-frequency
component."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    # Explicit softmax attention: unlike the fused SDPA kernels, it supports
    # the second-order gradients required by the CFBL objective.
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTTiny(nn.Module):
    def __init__(self, in_channels, embed_dim=192, depth=6, num_heads=3,
                 patch_size=8, base_grid=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, base_grid, base_grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_channels = embed_dim

    def forward(self, x):
        x = self.patch_embed(x)                     # (B, C, h, w)
        b, c, h, w = x.shape
        pos = F.interpolate(self.pos_embed, size=(h, w),
                            mode="bicubic", align_corners=False)
        x = (x + pos).flatten(2).transpose(1, 2)    # (B, h*w, C)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(b, c, h, w)
