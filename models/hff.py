"""Hybrid-Frequency Fusion (HFF).

Fuses two heterogeneous feature maps through channel-wise attention and
cross-branch channel interaction (Sec. 3.4 of the paper). Per-branch channel
attentions are computed from pooled features (Eq. 12), correlated into a
cross-branch attention matrix (Eq. 13), refined through dimension-matching
MLPs (Eq. 14), and the re-weighted branches are projected to a common
channel dimension and summed (Eq. 15).

HFF is applied twice in F2Net: first over the two low-frequency sub-branch
outputs (F_s, F_l) -> F_sl, then over (F_sl, F_m) with the high-frequency
branch output.

The two dimension-matching MLPs of Eq. 14 map the cross-branch matrix M
(Cs x Cl) back to each branch's channel dimension row-wise (M -> R^Cs) and
column-wise (M^T -> R^Cl), preserving per-channel identity for the addition
with the original attentions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channel_mlp(channels, reduction):
    hidden = max(channels // reduction, 8)
    return nn.Sequential(
        nn.Linear(channels, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, channels),
    )


def _cross_mlp(in_dim, reduction):
    hidden = max(in_dim // reduction, 8)
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 1),
    )


class HybridFrequencyFusion(nn.Module):
    def __init__(self, in_channels_a, in_channels_b, out_channels, reduction=4):
        super().__init__()
        # Eq. 12: channel attention per branch.
        self.att_a = _channel_mlp(in_channels_a, reduction)
        self.att_b = _channel_mlp(in_channels_b, reduction)

        # Eq. 14: two separate MLPs mapping the cross-branch matrix M back to
        # each branch's channel dimension (row-/column-wise for matching).
        self.cross_a = _cross_mlp(in_channels_b, reduction)
        self.cross_b = _cross_mlp(in_channels_a, reduction)

        # Eq. 15: two separate 1x1 convolutions project the re-weighted
        # branches into a unified channel dimension before summation.
        self.proj_a = nn.Conv2d(in_channels_a, out_channels, kernel_size=1)
        self.proj_b = nn.Conv2d(in_channels_b, out_channels, kernel_size=1)

    def forward(self, feat_a, feat_b):
        if feat_b.shape[2:] != feat_a.shape[2:]:
            feat_b = F.interpolate(feat_b, feat_a.shape[2:], mode="bilinear",
                                   align_corners=False)

        # Eq. 12: A = sigmoid(MLP(Pool(F))).
        att_a = torch.sigmoid(self.att_a(feat_a.mean(dim=(2, 3))))   # (B, Ca)
        att_b = torch.sigmoid(self.att_b(feat_b.mean(dim=(2, 3))))   # (B, Cb)

        # Eq. 13: cross-branch relationship matrix M = sigmoid(A_a A_b^T).
        m = torch.sigmoid(att_a.unsqueeze(2) @ att_b.unsqueeze(1))   # (B, Ca, Cb)

        # Eq. 14: refined attentions.
        ref_a = torch.sigmoid(self.cross_a(m).squeeze(-1) + att_a)             # (B, Ca)
        ref_b = torch.sigmoid(self.cross_b(m.transpose(1, 2)).squeeze(-1) + att_b)  # (B, Cb)

        # Eq. 15: F = Conv(A_a . F_a) + Conv(A_b . F_b).
        fused = self.proj_a(feat_a * ref_a[..., None, None]) \
              + self.proj_b(feat_b * ref_b[..., None, None])
        return fused
