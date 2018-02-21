import cv2
import numpy as np


def resize_image(image, size, channel_first = False):
    if np.ndim(image) == 3 and channel_first:
        image = image.transpose(1, 2, 0)

    if isinstance(size, int):
        image = cv2.resize(image, (size, size))
    else:
        image = cv2.resize(image, size)

    if np.ndim(image) == 3 and channel_first:
        image = image.transpose(2, 0, 1)
    return np.array(image)


def combine_images(images, num_columns):
    assert len(images) % num_columns == 0
    return np.concatenate([np.concatenate(images[k::num_columns], 0) for k in range(num_columns)], 1)
