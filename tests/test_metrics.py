import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import ConfusionMatrix


def test_hand_computed_case():
    cm = ConfusionMatrix(num_classes=3, ignore_index=255)
    cm.update(np.array([0, 0, 1, 1, 2, 2, 1]),
              np.array([0, 1, 1, 1, 2, 0, 255]))   # last pixel ignored
    r = cm.results(exclude=(0,))
    # class 1: tp=2 fp=0 fn=1 -> IoU 2/3, F1 0.8
    # class 2: tp=1 fp=1 fn=0 -> IoU 1/2, F1 2/3
    assert abs(r["IoU"][1] - 2 / 3) < 1e-9
    assert abs(r["IoU"][2] - 1 / 2) < 1e-9
    assert abs(r["mF1"] - (0.8 + 2 / 3) / 2) < 1e-9
    assert abs(r["aAcc"] - 4 / 6) < 1e-9


def test_perfect_prediction():
    cm = ConfusionMatrix(num_classes=2)
    label = np.random.randint(0, 2, (32, 32))
    cm.update(label, label)
    r = cm.results()
    assert r["mIoU"] == 1.0 and r["mF1"] == 1.0 and r["aAcc"] == 1.0
