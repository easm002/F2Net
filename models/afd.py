"""Adaptive Frequency Decomposition (AFD).

Decomposes the input image into low- and high-frequency components with
spatially-adaptive, content-conditioned low-pass filters (Sec. 3.2 of the
paper). A pointwise convolution first lifts the image into a feature space
(Eq. 2); the features are split into N channel groups (Eq. 3); each group
predicts its own per-pixel low-pass kernel via a softmax-normalized
convolution (Eq. 4); the high-pass response is the complement of the
low-pass one (Eq. 5); both filters are applied depthwise (Eq. 6) and the
groups are concatenated back (Eq. 7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFrequencyDecomposition(nn.Module):
    def __init__(self, in_channels=3, embed_dim=32, groups=8, kernel_size=3):
        super().__init__()
        assert embed_dim % groups == 0, "embed_dim must be divisible by groups"
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.groups = groups
        self.kernel_size = kernel_size
        self.group_dim = embed_dim // groups

        # Eq. 2: pointwise convolution for cross-channel understanding.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Eq. 4: one filter-generating convolution per channel group,
        # predicting a k*k low-pass kernel at every spatial position.
        self.filter_gen = nn.ModuleList([
            nn.Conv2d(self.group_dim, kernel_size ** 2, kernel_size=3, padding=1)
            for _ in range(groups)
        ])

    def forward(self, x):
        _, _, h, w = x.shape
        k = self.kernel_size
        pad = k // 2

        x = self.proj(x)                                   # Eq. 2
        groups = x.chunk(self.groups, dim=1)               # Eq. 3

        low, high = [], []
        for xn, gen in zip(groups, self.filter_gen):
            # Eq. 4: softmax over the k^2 dimension yields valid low-pass
            # weights (non-negative, summing to one) at each position.
            w_lf = gen(xn).softmax(dim=1)                  # (B, k^2, H, W)

            # Eq. 6: apply the per-pixel kernels depthwise. Rather than
            # materializing all k^2 shifted copies at once (unfold), the
            # k^2 taps are accumulated from views into one padded tensor,
            # which keeps peak memory at O(H*W) even for UHR inputs.
            xn_pad = F.pad(xn, (pad, pad, pad, pad))
            x_lf = xn.new_zeros(xn.shape)
            for idx in range(k * k):
                di, dj = divmod(idx, k)
                x_lf = x_lf + xn_pad[:, :, di:di + h, dj:dj + w] \
                            * w_lf[:, idx:idx + 1]

            # Eq. 5: the high-pass kernel is the identity kernel minus the
            # low-pass kernel, i.e. the high-pass response is the residual.
            x_hf = xn - x_lf

            low.append(x_lf)
            high.append(x_hf)

        # Eq. 7: reconstruct via channel-wise concatenation.
        return torch.cat(low, dim=1), torch.cat(high, dim=1)
