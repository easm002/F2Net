# F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation

> **News (March 2026):** 🎉 Our paper has been accepted by **CVPR 2026**!

Official repository of the CVPR 2026 paper **"F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation"**.

This repository provides reference implementations of the core components of F2Net:

| Component | Paper | Code |
|---|---|---|
| Adaptive Frequency Decomposition (AFD) | Sec. 3.2, Eq. 2–7 | `models/afd.py` |
| High-frequency encoder (VMamba-Tiny-M2) | Sec. 3.2 / 4.1 | `models/vmamba/` |
| Hybrid-Frequency Fusion (HFF) | Sec. 3.4, Eq. 12–15 | `models/hff.py` |
| Cross-frequency objectives (CFAL / CFBL) | Sec. 3.5, Eq. 16–17 | `models/losses.py` |

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
- [ISDNet](https://github.com/cedricgsh/ISDNet) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) for the UHR segmentation problem setting.
