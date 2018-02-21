from __future__ import print_function

import os
import shutil


def mkdir(path, clean = False):
    if clean:
        rm(path)

    if isinstance(path, (list, tuple)):
        for name in path:
            mkdir(name)

    if not os.path.exists(path):
        os.makedirs(path)


def rm(path):
    if isinstance(path, (list, tuple)):
        for name in path:
            rm(name)

    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def mv(src, dest):
    shutil.move(src, dest)


def cp(src, dest):
    shutil.copy2(src, dest)


def run(command, verbose = False):
    def parse(args):
        if isinstance(args, str):
            return args
        elif isinstance(args, (list, tuple)):
            return ' '.join([parse(arg) for arg in args])
        else:
            return str(args)

    command = parse(command)
    if verbose:
        print('==>', command)
    else:
        command += ' > /dev/null 2>&1'

    return os.system(command)
