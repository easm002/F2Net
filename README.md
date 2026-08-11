# F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation

> **News (March 2026):** 🎉 Our paper has been accepted by **CVPR 2026**!

Official repository of the CVPR 2026 paper **"F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation"**.

| Component | Paper | Code |
|---|---|---|
| Adaptive Frequency Decomposition (AFD) | Sec. 3.2, Eq. 2–7 | `models/afd.py` |
| High-frequency branch (VMamba-Tiny-M2) | Sec. 3.2 / 4.1 | `models/high_frequency.py`, `models/vmamba/` |
| Short-range sub-branch (DeepLabv3, ResNet-18) | Sec. 3.3 / 4.1 | `models/deeplabv3.py` |
| Long-range sub-branch (6-layer ViT-tiny) | Sec. 3.3 / 4.1 | `models/vit.py` |
| Hybrid-Frequency Fusion (HFF) | Sec. 3.4, Eq. 12–15 | `models/hff.py` |
| Cross-frequency objectives (CFAL / CFBL) | Sec. 3.5, Eq. 16–17 | `models/losses.py` |
| Evaluation metrics (mIoU / F1) | Sec. 4.1 | `utils/metrics.py` |
| Dataset preparation (DeepGlobe, Inria Aerial) | Sec. 4.1 | `datasets/` |

## Data preparation

[DeepGlobe Land Cover](http://deepglobe.org/) (803 × 2448×2448, 7 classes) and the
[Inria Aerial Image Labeling benchmark](https://project.inria.fr/aerialimagelabeling/)
(180 × 5000×5000, binary building masks) are prepared with:

```bash
python datasets/prepare_deepglobe.py --src /path/to/land-train --out data/deepglobe
python datasets/prepare_inria.py --src /path/to/AerialImageDataset/train --out data/inria
```

The scripts convert the color-coded masks to label maps and apply the community
splits (455/207/142 and 126/27/27, following GLNet / FCtL / ISDNet).

## Citation

```bibtex
@inproceedings{chen2026f2net,
  title={F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation},
  author={Chen, Hengzhi and Feng, Liqian and Wu, Wenhua and Zhu, Xiaogang and Wu, Qiuxia and Shan, Lianlei and Hu, Kun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## Acknowledgements

- [VMamba](https://github.com/MzeroMiko/VMamba) — `models/vmamba/` is adapted from the official VMamba implementation (MIT License); its `mamba2/` ops originate from [mamba](https://github.com/state-spaces/mamba) (Apache-2.0).
- [ISDNet](https://github.com/cedricgsh/ISDNet), [GLNet](https://github.com/VITA-Group/GLNet) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for the UHR segmentation problem setting and evaluation protocol.
