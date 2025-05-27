from scipy import io as sio


def load_false_images_stem():
    false_images_mat = sio.loadmat("./dataset/colorchecker_illuminant/falseimages.mat")
    false_images_array = false_images_mat["falseimages"][:, 1]
    false_images_stem = [str(fname_obj[0]) for fname_obj in false_images_array]
    return false_images_stem
