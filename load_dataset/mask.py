import cv2
import numpy as np

from load_dataset.colorchecker import get_selected_patch


def create_gray_mask(corrected_im: np.ndarray, coords) -> np.ndarray:
    # procedure
    # 1. Convert the image to Lab color space
    # 2. Set the L channel to a constant value (e.g., 5)
    # 3. Calculate mean and standard deviation of a and b channels in the gray patches
    # 4. Create masks for upper and lower bounds of a and b channels
    # 5. Combine the masks to create a final gray mask
    # 6. Apply median blur to the gray mask
    # 7. Convert the mask to binary format

    corrected_lab = cv2.cvtColor(corrected_im, cv2.COLOR_BGR2Lab)
    corrected_lab[..., 0] = 5

    colors = [18, 19, 20, 21, 22, 23]
    mask = np.zeros(corrected_im.shape[:2], dtype=np.float32)
    for color in colors:
        mask += get_selected_patch(corrected_im, coords, color)

    gray_region = corrected_lab * mask[..., None]
    gray_a = gray_region[..., 1]
    gray_a = gray_a[mask != 0]
    gray_b = gray_region[..., 2]
    gray_b = gray_b[mask != 0]

    threshold = 1.5
    lab = corrected_lab.copy()
    im_a = lab[..., 1]
    upper_mask_a = im_a < (gray_a.mean() + threshold * gray_a.std())
    lower_mask_a = im_a > (gray_a.mean() - threshold * gray_a.std())

    im_b = lab[..., 2]
    upper_mask_b = im_b < (gray_b.mean() + threshold * gray_b.std())
    lower_mask_b = im_b > (gray_b.mean() - threshold * gray_b.std())

    upper_mask = upper_mask_a & upper_mask_b
    lower_mask = lower_mask_a & lower_mask_b
    gray_mask = upper_mask & lower_mask
    gray_mask = gray_mask.astype(np.uint8)
    gray_mask = cv2.medianBlur(gray_mask, 7)
    gray_mask = gray_mask > 0.5
    # binary mask
    gray_mask = gray_mask.astype(np.uint8) * 255

    return gray_mask
