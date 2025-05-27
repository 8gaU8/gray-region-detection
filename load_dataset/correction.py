from pathlib import Path

import numpy as np

from .colorchecker import get_cc_coords, get_selected_patch
from .images import load_raw


def load_white_corrected(path: Path) -> np.ndarray:
    # apply white balance correction with color checker
    image = load_raw(path)
    h, w, _c = image.shape
    coords = get_cc_coords(path, (w, h))
    image = image.astype(np.float32)
    white_mask = get_selected_patch(image, coords, 18)
    white_level = np.mean(image[white_mask > 0], axis=0)
    black_mask = get_selected_patch(image, coords, 23)
    black_level = np.min(image[black_mask > 0], axis=(0, 1))

    corrected_im = (image - black_level) / (white_level - black_level)
    return corrected_im
