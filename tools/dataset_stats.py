"""Per-class pixel statistics of a prepared dataset split.

Usage:
    python tools/dataset_stats.py --ann-dir data/deepglobe/ann_dir/train --num-classes 7
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    args = parser.parse_args()

    counts = np.zeros(args.num_classes, dtype=np.int64)
    files = sorted(args.ann_dir.glob("*.png"))
    assert files, f"no *.png masks under {args.ann_dir}"
    for path in files:
        ids = np.asarray(Image.open(path))
        counts += np.bincount(ids.reshape(-1), minlength=args.num_classes)[:args.num_classes]

    total = counts.sum()
    print(f"{len(files)} masks, {total} labeled pixels")
    print(f"{'class':>6} {'pixels':>15} {'share':>8}")
    for cls, count in enumerate(counts):
        print(f"{cls:>6} {count:>15} {count / max(total, 1):>8.2%}")


if __name__ == "__main__":
    main()
