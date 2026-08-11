"""Colorize a single-channel label map for visual inspection, optionally
blended over the corresponding image.

Usage:
    python tools/visualize.py --mask xxx_mask.png --dataset deepglobe --out vis.png
    python tools/visualize.py --mask xxx_mask.png --dataset inria --image xxx_sat.tif --alpha 0.5 --out vis.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PALETTES = {
    "deepglobe": [
        (0, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 0), (0, 0, 255), (255, 255, 255),
    ],
    "inria": [(0, 0, 0), (255, 255, 255)],
}


def colorize(ids, palette):
    lut = np.zeros((256, 3), dtype=np.uint8)
    for cls, color in enumerate(palette):
        lut[cls] = color
    return lut[ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True,
                        help="single-channel label map (class ids)")
    parser.add_argument("--dataset", choices=sorted(PALETTES), required=True)
    parser.add_argument("--image", type=Path, default=None,
                        help="optional image to blend the colorized mask over")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ids = np.asarray(Image.open(args.mask))
    color = colorize(ids, PALETTES[args.dataset])

    if args.image is not None:
        image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float64)
        color = (args.alpha * color + (1 - args.alpha) * image).astype(np.uint8)

    Image.fromarray(color).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
