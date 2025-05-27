from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from load_dataset.colorchecker import fill_colorchecker, get_cc_coords
from load_dataset.correction import load_white_corrected
from load_dataset.false_images import load_false_images_stem
from load_dataset.images import load_img
from load_dataset.mask import create_gray_mask


def mk_dataset():
    false_images_stem = load_false_images_stem()
    root = Path("./dataset/all_images")

    label_root = Path("./dataset/dataset/labels")
    if not label_root.exists():
        label_root.mkdir(parents=True)

    img_root = Path("./dataset/dataset/images")
    if not img_root.exists():
        img_root.mkdir(parents=True)

    sorted_path_list = sorted(list(root.glob("*.png")))

    for path in tqdm(sorted_path_list):
        if path.stem in false_images_stem:
            continue
        label_path = label_root / path.name
        img_path = img_root / path.name
        org_im = load_img(path)
        org_im /= org_im.max()
        h, w, c = org_im.shape
        coords = get_cc_coords(path, (w, h))
        org_im = fill_colorchecker(org_im, coords)
        org_im *= 255
        org_im = org_im.astype(np.uint8)
        cv2.imwrite(str(img_path), org_im)
        del org_im

        corrected_im = load_white_corrected(path)
        gray_mask = create_gray_mask(corrected_im, coords)
        gray_mask = fill_colorchecker(gray_mask, coords)
        gray_mask = gray_mask.astype(np.uint8)
        cv2.imwrite(str(label_path), gray_mask)


def main():
    mk_dataset()


if __name__ == "__main__":
    main()
