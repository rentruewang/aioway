# Copyright (c) AIoWay Authors - All Rights Reserved

import os

from torchvision import io as vio


def read_image(fname: os.PathLike[str]):
    fname_str = str(fname)
    return vio.decode_image(fname_str)


def read_image_normalized(fname: os.PathLike[str]):
    return read_image(fname).float() / 255
