"""Prepare the Inria Aerial Image Labeling dataset.

Converts the 180 annotated training tiles (5 cities x 36 tiles, 5000x5000)
into binary label maps and divides them into train/val/test = 126/27/27,
the protocol used by FCtL and ISDNet. The split is stratified by city.

Usage:
    python prepare_inria.py --src /path/to/AerialImageDataset/train --out /path/to/inria

``--src`` must contain ``images/*.tif`` and ``gt/*.tif``. To reproduce a
specific published division exactly, pass ``--lists DIR`` with
train.txt / val.txt / test.txt of tile names (without extension);
otherwise a deterministic seeded per-city split is generated.

Output layout (mmsegmentation-style):
    out/img_dir/{train,val,test}/<tile>_sat.tif
    out/ann_dir/{train,val,test}/<tile>_mask.png   (0 background, 1 building)
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None   # Inria tiles are 5000x5000


def seeded_city_split(ids, fractions, seed):
    """City-stratified deterministic split with exact global counts:
    tiles are shuffled within each city, interleaved across cities
    round-robin, and the interleaved list is sliced by the global
    val/test counts."""
    by_city = defaultdict(list)
    for tile in ids:
        city = re.match(r"([a-zA-Z-]+)", tile).group(1)
        by_city[city].append(tile)

    rng = random.Random(seed)
    queues = []
    for city in sorted(by_city):
        tiles = sorted(by_city[city])
        rng.shuffle(tiles)
        queues.append(tiles)

    interleaved = []
    while any(queues):
        for queue in queues:
            if queue:
                interleaved.append(queue.pop())

    n_val = round(len(ids) * fractions["val"])
    n_test = round(len(ids) * fractions["test"])
    return {
        "val": interleaved[:n_val],
        "test": interleaved[n_val:n_val + n_test],
        "train": interleaved[n_val + n_test:],
    }


def read_lists(lists_dir, ids):
    splits = {}
    for name in ("train", "val", "test"):
        listed = (lists_dir / f"{name}.txt").read_text().split()
        missing = sorted(set(listed) - set(ids))
        assert not missing, f"{name}.txt tiles not found in --src: {missing[:5]}"
        splits[name] = listed
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="AerialImageDataset/train with images/ and gt/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lists", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ids = sorted(p.stem for p in (args.src / "images").glob("*.tif"))
    assert ids, f"no images/*.tif found under {args.src}"

    fractions = {"val": 27 / 180, "test": 27 / 180}
    splits = read_lists(args.lists, ids) if args.lists \
        else seeded_city_split(ids, fractions, args.seed)

    for name, split_ids in splits.items():
        img_dir = args.out / "img_dir" / name
        ann_dir = args.out / "ann_dir" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for tile in split_ids:
            shutil.copy(args.src / "images" / f"{tile}.tif",
                        img_dir / f"{tile}_sat.tif")
            gt = np.asarray(Image.open(args.src / "gt" / f"{tile}.tif"))
            Image.fromarray((gt > 127).astype(np.uint8)).save(
                ann_dir / f"{tile}_mask.png")
        print(f"{name}: {len(split_ids)} tiles")


if __name__ == "__main__":
    main()
