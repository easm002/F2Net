# Dataset preparation

## DeepGlobe Land Cover

Download the Land Cover Classification data from the
[DeepGlobe challenge](http://deepglobe.org/) (requires registration).
The annotated `land-train` set contains 803 satellite images
(2448×2448) with RGB-coded masks:

```
land-train/
├── 100694_sat.jpg
├── 100694_mask.png
└── ...
```

Convert and split (train/val/test = 455/207/142, the community protocol of
GLNet / ISDNet):

```bash
python datasets/prepare_deepglobe.py --src /path/to/land-train --out data/deepglobe
```

| id | class | mask color |
|---|---|---|
| 0 | unknown (ignored) | (0, 0, 0) |
| 1 | urban | (0, 255, 255) |
| 2 | agriculture | (255, 255, 0) |
| 3 | rangeland | (255, 0, 255) |
| 4 | forest | (0, 255, 0) |
| 5 | water | (0, 0, 255) |
| 6 | barren | (255, 255, 255) |

## Inria Aerial Image Labeling

Download from the [official site](https://project.inria.fr/aerialimagelabeling/).
The annotated `train` set contains 180 tiles (5000×5000) over five cities:

```
AerialImageDataset/train/
├── images/austin1.tif ...
└── gt/austin1.tif ...
```

Convert and split (train/val/test = 126/27/27, following FCtL / ISDNet;
city-stratified):

```bash
python datasets/prepare_inria.py --src /path/to/AerialImageDataset/train --out data/inria
```

Both scripts accept `--lists DIR` (train.txt / val.txt / test.txt of image
ids) to reproduce a specific published division exactly, and write an
mmsegmentation-style layout:

```
data/<dataset>/
├── img_dir/{train,val,test}/
└── ann_dir/{train,val,test}/
```

## Utilities

```bash
# per-class pixel statistics of a prepared dataset
python tools/dataset_stats.py --ann-dir data/deepglobe/ann_dir/train --num-classes 7

# colorize a predicted/ground-truth label map for inspection
python tools/visualize.py --mask data/deepglobe/ann_dir/val/xxx_mask.png --dataset deepglobe --out vis.png
```
