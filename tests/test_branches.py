import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from deeplabv3 import DeepLabV3
from vit import ViTTiny


def test_deeplabv3_output_stride_8():
    model = DeepLabV3(in_channels=3, channels=128)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 128, 8, 8)


def test_deeplabv3_batch_one_training():
    model = DeepLabV3(in_channels=3).train()
    model(torch.randn(1, 3, 64, 64)).mean().backward()


def test_vit_tiny_geometry():
    model = ViTTiny(in_channels=3)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 192, 8, 8)      # patch size 8
    assert len(model.blocks) == 6           # 6-layer ViT-tiny


def test_vit_variable_input_size():
    model = ViTTiny(in_channels=3).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 96, 40))
    assert out.shape == (1, 192, 12, 5)
