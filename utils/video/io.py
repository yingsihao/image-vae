import os

from ..image import load_image, save_image
from ..shell import mkdir, rm, run


def load_video(path, clean = True):
    images_path = '{0}.images'.format(path)
    mkdir(images_path, clean = True)

    run(('ffmpeg', ('-i', path), os.path.join(images_path, '%d.png')))
    num_images = len(os.listdir(images_path))
    images = [load_image(os.path.join(images_path, '{0}.png'.format(k + 1))) for k in range(num_images)]

    if clean:
        rm(images_path)
    return images


def save_video(images, path, clean = True):
    images_path = '{0}.images'.format(path)
    mkdir(images_path, clean = True)

    rm(path)
    for k, image in enumerate(images):
        save_image(image, os.path.join(images_path, '{0}.png'.format(k + 1)))
    run(('ffmpeg', ('-r', 60), ('-f', 'image2'), ('-s', '1920x1080'),
         ('-i', os.path.join(images_path, '%d.png')), ('-vcodec', 'libx264'),
         ('-crf', 25), ('-pix_fmt', 'yuv420p'), path))

    if clean:
        rm(images_path)
