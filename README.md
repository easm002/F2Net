# F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation

> **News (March 2026):** 🎉 Our paper has been accepted by **CVPR 2026**!

Official repository of the CVPR 2026 paper **"F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation"**.

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
