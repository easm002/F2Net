import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from hff import HybridFrequencyFusion


def test_fuses_mismatched_channels_and_resolution():
    hff = HybridFrequencyFusion(in_channels_a=128, in_channels_b=512,
                                out_channels=128)
    feat_a = torch.randn(2, 128, 32, 32)
    feat_b = torch.randn(2, 512, 8, 8)      # coarser branch is upsampled
    fused = hff(feat_a, feat_b)
    assert fused.shape == (2, 128, 32, 32)


def test_gradients_reach_both_branches():
    hff = HybridFrequencyFusion(64, 96, 32)
    feat_a = torch.randn(1, 64, 16, 16, requires_grad=True)
    feat_b = torch.randn(1, 96, 16, 16, requires_grad=True)
    hff(feat_a, feat_b).mean().backward()
    assert feat_a.grad is not None and feat_a.grad.abs().sum() > 0
    assert feat_b.grad is not None and feat_b.grad.abs().sum() > 0
