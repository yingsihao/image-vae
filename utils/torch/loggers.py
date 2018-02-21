from __future__ import print_function

import os
from collections import OrderedDict

import tensorflow as tf

from ..image import save_image, save_images
from ..shell import mkdir


class Logger:
    def __init__(self, save_path, num_steps = 16, refresh = 60):
        self.save_path = save_path
        self.num_steps = num_steps
        self.refresh = refresh
        self.writer = tf.summary.FileWriter(self.save_path)
        self.image_paths = {}

    def scalar_summary(self, name, value, step):
        self.writer.add_summary(tf.Summary(value = [tf.Summary.Value(tag = name, simple_value = value)]), step)
        self.writer.flush()

    def image_summary(self, name, images, step, channel_first = True):
        web_path = os.path.join(self.save_path, 'web')
        images_path = os.path.join('images', '{0}-{1}'.format(name, step))
        mkdir(os.path.join(web_path, images_path), clean = True)

        # save
        for k, image in enumerate(images):
            if isinstance(image, (list, tuple)):
                image_path = os.path.join(images_path, '{0}-{1}-{2}.gif'.format(name, step, k))
                save_images(image, os.path.join(web_path, image_path), channel_first = channel_first)
            else:
                image_path = os.path.join(images_path, '{0}-{1}-{2}.png'.format(name, step, k))
                save_image(image, os.path.join(web_path, image_path), channel_first = channel_first)

            if step not in self.image_paths:
                self.image_paths[step] = OrderedDict()
            if name not in self.image_paths[step]:
                self.image_paths[step][name] = []

            self.image_paths[step][name].append(image_path)

        # visualize
        with open(os.path.join(web_path, 'index.html'), 'w') as fp:
            print('<meta http-equiv="refresh" content="{0}">'.format(self.refresh), file = fp)
            for step in sorted(self.image_paths.keys(), reverse = True)[:self.num_steps]:
                print('<h3>step [{0}]</h3>'.format(step), file = fp)

                # table
                print('<table border="1" style="table-layout: fixed;">', file = fp)
                for name in self.image_paths[step].keys():
                    print('<tr>', file = fp)
                    for image_path in self.image_paths[step][name]:
                        print('<td halign="center" style="word-wrap: break-word;" valign="top"><p>', file = fp)
                        print('<img src="{1}" style="width:128px;"><br><p>{0}</p>'.format(name, image_path), file = fp)
                        print('</p></td>', file = fp)
                    print('</tr>', file = fp)
                print('</table>', file = fp)
