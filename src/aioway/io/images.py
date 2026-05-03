# Copyright (c) AIoWay Authors - All Rights Reserved

from torchvision import io as vio


def read_image(fname: str):
    return vio.decode_image(fname)


def read_image_normalized(fname: str):
    return read_image(fname).float() / 255
