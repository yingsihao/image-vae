from __future__ import print_function

import logging
import os

thisfile = os.path.abspath(__file__)

try:
    from . import bullet
except ImportError:
    logging.error('{}: failed to import "bullet"'.format(thisfile))

try:
    from . import image
except ImportError:
    logging.error('{}: failed to import "image"'.format(thisfile))

try:
    from . import shell
except ImportError:
    logging.error('{}: failed to import "shell"'.format(thisfile))

try:
    from . import torch
except ImportError:
    logging.error('{}: failed to import "torch"'.format(thisfile))

try:
    from . import video
except ImportError:
    logging.error('{}: failed to import "video"'.format(thisfile))

try:
    from . import math
except ImportError:
    logging.error('{}: failed to import "math"'.format(thisfile))

try:
    from .misc import *
except ImportError:
    logging.error('{}: failed to import "misc"'.format(thisfile))
