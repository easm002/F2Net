import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from afd import AdaptiveFrequencyDecomposition


def test_output_shapes():
    afd = AdaptiveFrequencyDecomposition(in_channels=3, embed_dim=32, groups=8)
    low, high = afd(torch.randn(2, 3, 65, 47))
    assert low.shape == (2, 32, 65, 47)
    assert high.shape == (2, 32, 65, 47)


def test_decomposition_is_conservative():
    # The low- and high-pass responses must sum back to the projected input
    # (the high-pass kernel is the identity minus the low-pass kernel).
    afd = AdaptiveFrequencyDecomposition().eval()
    x = torch.randn(1, 3, 40, 40)
    with torch.no_grad():
        low, high = afd(x)
        assert torch.allclose(low + high, afd.proj(x), atol=1e-5)


def test_lowpass_weights_are_normalized():
    afd = AdaptiveFrequencyDecomposition().eval()
    x = afd.proj(torch.randn(1, 3, 16, 16))
    w = afd.filter_gen[0](x[:, :afd.group_dim]).softmax(dim=1)
    assert torch.allclose(w.sum(dim=1), torch.ones(1, 16, 16), atol=1e-5)


def test_even_kernel_rejected():
    with pytest.raises(AssertionError):
        AdaptiveFrequencyDecomposition(kernel_size=4)


def test_gradients_flow():
    afd = AdaptiveFrequencyDecomposition()
    low, high = afd(torch.randn(1, 3, 24, 24))
    (low.mean() + high.mean()).backward()
    assert afd.proj.weight.grad is not None
    assert afd.filter_gen[0].weight.grad is not None
