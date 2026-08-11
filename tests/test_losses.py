import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from losses import CrossFrequencyAlignmentLoss, cross_frequency_balance_loss


def test_cfal_scalar_and_differentiable():
    cfal = CrossFrequencyAlignmentLoss(channels_sl=64, channels_m=128)
    f_sl = torch.randn(2, 64, 16, 16, requires_grad=True)
    f_m = torch.randn(2, 128, 8, 8)
    loss = cfal(f_sl, f_m)
    assert loss.dim() == 0 and loss.item() >= 0
    loss.backward()
    assert f_sl.grad is not None


def test_cfal_zero_for_identical_distributions():
    cfal = CrossFrequencyAlignmentLoss(channels_sl=32, channels_m=32, embed_dim=32)
    # Force both projections to the identity so the two inputs project to
    # the same logits -> the symmetric KL must vanish.
    with torch.no_grad():
        for proj in (cfal.proj_sl, cfal.proj_m):
            proj.weight.copy_(torch.eye(32).view(32, 32, 1, 1))
            proj.bias.zero_()
    x = torch.randn(1, 32, 8, 8)
    assert cfal(x, x.clone()).item() < 1e-6


def test_cfbl_balances_toy_branches():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Linear(8, 8)
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            return self.head(torch.cat([self.a(x), self.b(x)], dim=-1))

    toy = Toy()
    logits = toy(torch.randn(4, 8))
    ce = F.cross_entropy(logits, torch.randint(0, 3, (4,)))
    cfbl = cross_frequency_balance_loss(
        ce, {"a": toy.a.parameters(), "b": toy.b.parameters()})
    assert cfbl.dim() == 0 and cfbl.item() >= 0
    (ce + 0.1 * cfbl).backward()          # differentiable second-order path
    assert toy.a.weight.grad is not None
