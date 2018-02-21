import imageio
import numpy as np

from .misc import resize_image

import warnings
def load_image(path, size = None, channel_first = False):
    image = imageio.imread(path).astype(np.float32) / 255.

    if size is not None:
        image = resize_image(image, size)

    if np.ndim(image) == 3 and channel_first:
        image = image.transpose(2, 0, 1)
    return np.array(image)


def save_image(image, path, channel_first = False):
    if np.ndim(image) == 3 and channel_first:
        image = image.transpose(1, 2, 0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        imageio.imwrite(path, image)


def save_images(images, path, duration = 1, channel_first = False):
    for k, image in enumerate(images):
        if np.ndim(image) == 3 and channel_first:
            images[k] = image.transpose(1, 2, 0)

    writer = imageio.get_writer(path, duration = duration)
    for image in images:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            writer.append_data(image)
    writer.close()
