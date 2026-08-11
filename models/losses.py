"""Cross-frequency training objectives (Sec. 3.5 of the paper).

- CrossFrequencyAlignmentLoss (CFAL, Eq. 16): symmetric KL divergence
  between the fused low-frequency representation F_sl and the
  high-frequency branch output F_m, enforcing semantic consistency across
  frequency branches.
- cross_frequency_balance_loss (CFBL, Eq. 17): regularizes the gradient
  magnitudes of the frequency branches so that no single branch dominates
  optimization.

The overall objective (Eq. 18) is
    L = lambda_1 * L_CFAL + lambda_2 * L_CFBL + lambda_3 * L_CE
with lambda_1 = lambda_2 = 0.1 and lambda_3 = 1 (Sec. 4.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrequencyAlignmentLoss(nn.Module):
    """CFAL (Eq. 16).

    The two branch outputs differ in channel width and resolution, so they
    are first projected to a shared embedding dimension and spatially
    aligned; each spatial position is then normalized into a channel
    distribution and the symmetric KL divergence is averaged over
    positions:

        L_CFAL = 1/2 [ D_KL(F_sl || F_m) + D_KL(F_m || F_sl) ].
    """

    def __init__(self, channels_sl, channels_m, embed_dim=128):
        super().__init__()
        self.proj_sl = nn.Conv2d(channels_sl, embed_dim, kernel_size=1)
        self.proj_m = nn.Conv2d(channels_m, embed_dim, kernel_size=1)

    def forward(self, f_sl, f_m):
        if f_m.shape[2:] != f_sl.shape[2:]:
            f_m = F.interpolate(f_m, f_sl.shape[2:], mode="bilinear",
                                align_corners=False)
        log_p = F.log_softmax(self.proj_sl(f_sl), dim=1)
        log_q = F.log_softmax(self.proj_m(f_m), dim=1)
        p, q = log_p.exp(), log_q.exp()
        kl_pq = (p * (log_p - log_q)).sum(dim=1).mean()
        kl_qp = (q * (log_q - log_p)).sum(dim=1).mean()
        return 0.5 * (kl_pq + kl_qp)


def cross_frequency_balance_loss(task_loss, branch_params):
    """CFBL (Eq. 17).

        L_CFBL = sum_Theta | G_Theta - mean(G) |,
        G_Theta = || grad_Theta L_CE ||_2

    Args:
        task_loss: the (scalar) segmentation cross-entropy loss.
        branch_params: dict mapping branch name -> iterable of that branch's
            parameters. Following common gradient-balancing practice
            (GradNorm), restricting each entry to the branch's final block
            keeps the second-order cost manageable.

    The gradient norms are computed with ``create_graph=True`` so the
    penalty itself is differentiable and can be added to the total loss.
    """
    norms = []
    for params in branch_params.values():
        grads = torch.autograd.grad(task_loss, list(params),
                                    create_graph=True, retain_graph=True,
                                    allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        norms.append(flat.norm(p=2))
    g = torch.stack(norms)
    return (g - g.mean()).abs().sum()
