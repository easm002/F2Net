"""High-frequency branch: a VMamba-Tiny-M2 visual state-space encoder over
the full-resolution high-frequency component (Sec. 3.2 / 4.1 of the paper):
four stages of VSS blocks with depths [2, 2, 4, 2] and a base channel
dimension of 64."""

try:
    from .vmamba import VSSM
except ImportError:
    from vmamba import VSSM

import torch.nn as nn


class HighFrequencyBranch(nn.Module):
    def __init__(self, in_channels, depths=(2, 2, 4, 2),
                 dims=(64, 128, 256, 512), drop_path_rate=0.2):
        super().__init__()
        self.vssm = VSSM(
            depths=list(depths), dims=list(dims), drop_path_rate=drop_path_rate,
            patch_size=4, in_chans=in_channels, num_classes=0,
            ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
            ssm_init="v2", forward_type="m0_noz",
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln",
            downsample_version="v3", patchembed_version="v2",
            use_checkpoint=False, posembed=False,
        )
        self.out_channels = dims[-1]

    def forward(self, x):
        x = self.vssm.patch_embed(x)
        for layer in self.vssm.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2).contiguous()   # (B, C, H/32, W/32)
