from pathlib import Path

import cv2
import numpy as np


def get_coord_path(im_path: Path) -> Path:
    fname = im_path.stem
    cc_path = f"./dataset/colorchecker_illuminant/ColorCheckerDatabase_MaskCoordinates/coordinates/{fname}_macbeth.txt"
    cc_path = Path(cc_path)
    if not cc_path.exists():
        raise FileNotFoundError(f"Colorchecker coordinates file not found: {cc_path}")
    return cc_path


def _line2float_list(line: str) -> list:
    return [float(x) for x in line.strip().split(" ")]


def get_cc_coords(im_path: Path, im_dim: tuple[int, int]) -> list:
    # load colorchecker coordinates
    im_cc_path = get_coord_path(im_path)
    with open(im_cc_path, "r") as f:
        coords = f.readlines()
    coords = [_line2float_list(line) for line in coords]

    # scale coordinates to image size
    coords = np.array(coords)
    dim = coords[0]
    scale = np.array(im_dim) / dim
    coords = coords * scale

    return coords


# plot points on image
def plot_colorchecker(im, coords):
    im0 = im.copy()
    for idx in range(5, coords.shape[0], 4):
        # idx =
        im0 = cv2.fillPoly(im0, [(coords[idx : idx + 4]).astype(np.int32)], (1, 0, 0))
    return im0


def get_selected_patch(im, coords, selected_color):
    selected_idx = 5 + selected_color * 4
    canvas = np.zeros(im.shape[:-1])
    mask = cv2.fillPoly(
        canvas, [(coords[selected_idx : selected_idx + 4]).astype(np.int32)], (1,)
    )
    return mask


def fill_colorchecker(im, coords):
    # fill the color checker to hide it
    polycoords = [
        coords[1],
        coords[2],
        coords[4],
        coords[3],
    ]
    polycoords = np.array(polycoords, dtype=np.int32)
    canvas = np.zeros(im.shape[:2])
    mask = cv2.fillPoly(canvas, [polycoords], (1,))
    im[mask == 1] = 0
    return im
