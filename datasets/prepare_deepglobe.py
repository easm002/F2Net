"""Prepare the DeepGlobe Land Cover dataset.

Converts the RGB-coded masks of the official `land-train` set (803 pairs of
``*_sat.jpg`` / ``*_mask.png`` at 2448x2448) into single-channel label maps
and divides them into train/val/test = 455/207/142, the community protocol
introduced by GLNet and followed by ISDNet and subsequent UHR work.

Usage:
    python prepare_deepglobe.py --src /path/to/land-train --out /path/to/deepglobe

To reproduce a specific published division exactly, pass ``--lists DIR``
where DIR contains train.txt / val.txt / test.txt with one image id per
line; otherwise a deterministic seeded split with the standard counts is
generated.

Output layout (mmsegmentation-style):
    out/img_dir/{train,val,test}/<id>_sat.jpg
    out/ann_dir/{train,val,test}/<id>_mask.png   (single-channel ids 0-6)

Classes: 0 unknown (ignored in evaluation), 1 urban, 2 agriculture,
3 rangeland, 4 forest, 5 water, 6 barren.
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# RGB color -> class id (DeepGlobe official coding).
CLASS_COLORS = {
    (0, 0, 0): 0,        # unknown
    (0, 255, 255): 1,    # urban
    (255, 255, 0): 2,    # agriculture
    (255, 0, 255): 3,    # rangeland
    (0, 255, 0): 4,      # forest
    (0, 0, 255): 5,      # water
    (255, 255, 255): 6,  # barren
}


def rgb_to_ids(mask_rgb):
    """Maps an (H, W, 3) RGB mask to (H, W) class ids. Channels are
    binarized at 128 first, as the official masks are 0/255-coded."""
    bits = (np.asarray(mask_rgb) > 127).astype(np.uint8)
    code = bits[..., 0] * 4 + bits[..., 1] * 2 + bits[..., 2]
    lut = np.zeros(8, dtype=np.uint8)
    for (r, g, b), cls in CLASS_COLORS.items():
        lut[(r > 127) * 4 + (g > 127) * 2 + (b > 127)] = cls
    return lut[code]


def read_lists(lists_dir, ids):
    splits = {}
    for name in ("train", "val", "test"):
        listed = (lists_dir / f"{name}.txt").read_text().split()
        missing = sorted(set(listed) - set(ids))
        assert not missing, f"{name}.txt ids not found in --src: {missing[:5]}"
        splits[name] = listed
    return splits


def seeded_split(ids, counts, seed):
    assert sum(counts.values()) == len(ids), \
        f"split counts {counts} do not sum to {len(ids)} images"
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    splits, start = {}, 0
    for name in ("train", "val", "test"):
        splits[name] = ids[start:start + counts[name]]
        start += counts[name]
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="directory with *_sat.jpg and *_mask.png")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lists", type=Path, default=None,
                        help="directory with train.txt/val.txt/test.txt image ids")
    parser.add_argument("--counts", type=int, nargs=3, default=(455, 207, 142),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ids = sorted(p.name[:-len("_sat.jpg")] for p in args.src.glob("*_sat.jpg"))
    assert ids, f"no *_sat.jpg found under {args.src}"

    counts = dict(zip(("train", "val", "test"), args.counts))
    splits = read_lists(args.lists, ids) if args.lists \
        else seeded_split(ids, counts, args.seed)

    for name, split_ids in splits.items():
        img_dir = args.out / "img_dir" / name
        ann_dir = args.out / "ann_dir" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for image_id in split_ids:
            shutil.copy(args.src / f"{image_id}_sat.jpg",
                        img_dir / f"{image_id}_sat.jpg")
            mask = Image.open(args.src / f"{image_id}_mask.png").convert("RGB")
            Image.fromarray(rgb_to_ids(mask)).save(
                ann_dir / f"{image_id}_mask.png")
        print(f"{name}: {len(split_ids)} images")


if __name__ == "__main__":
    main()
