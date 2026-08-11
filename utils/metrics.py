"""Segmentation evaluation metrics: per-class IoU, mIoU and F1, computed
from an accumulated confusion matrix (standard mmsegmentation-style
protocol). Classes marked with ``ignore_index`` are excluded."""

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, label):
        """Accumulates one prediction/label pair (integer class maps of the
        same shape; numpy arrays or torch tensors)."""
        pred = np.asarray(pred).reshape(-1)
        label = np.asarray(label).reshape(-1)
        valid = (label != self.ignore_index) & (label < self.num_classes)
        idx = label[valid] * self.num_classes + pred[valid]
        self.mat += np.bincount(idx, minlength=self.num_classes ** 2) \
                        .reshape(self.num_classes, self.num_classes)

    def results(self, exclude=()):
        """Returns per-class IoU / F1 and their means, excluding the class
        indices in ``exclude`` (e.g. an 'unknown' class) from the means."""
        tp = np.diag(self.mat).astype(np.float64)
        fp = self.mat.sum(axis=0) - tp
        fn = self.mat.sum(axis=1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0,
                      2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
        keep = np.array([c for c in range(self.num_classes) if c not in exclude])
        return {
            "IoU": iou,
            "F1": f1,
            "mIoU": float(np.nanmean(iou[keep])),
            "mF1": float(np.nanmean(f1[keep])),
            "aAcc": float(tp.sum() / max(self.mat.sum(), 1)),
        }
