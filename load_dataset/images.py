from pathlib import Path
import cv2
import numpy as np

MAX_PIXEL_VALUE = 2**12 - 1
BLACK_LEVEL_5D = 129


def load_raw(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return im


def preprocess_img(im, use_5d) -> np.ndarray:
    im = im.astype(np.float32)
    if use_5d:
        # black level correction
        im = im - BLACK_LEVEL_5D
        # clip negative values to 0
        im[np.less(im, 0)] = 0
    # normalize to [0, 1]
    im = im / MAX_PIXEL_VALUE
    return im


# read the data from the file
def load_img(path: Path) -> np.ndarray:
    use_5d = False
    if path.stem.startswith("IMG"):
        use_5d = True
    im = load_raw(path)
    im = preprocess_img(im, use_5d)
    return im