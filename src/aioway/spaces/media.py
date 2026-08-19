# Copyright (c) AIoWay Authors - All Rights Reserved


import torch
from torchrl.data import tensor_specs as tspecs

from .gym import scalar_box_space, unbounded_box_space

__all__ = ["float_image_space", "byte_image_space", "long_image_space"]


def float_image_space(num_channels: int, width: int, height: int) -> tspecs.Bounded:
    return scalar_box_space(0, 1, shape=torch.Size([num_channels, width, height]))


def byte_image_space(num_channels: int, width: int, height: int) -> tspecs.Unbounded:
    return unbounded_box_space(
        shape=torch.Size([num_channels, width, height]), dtype=torch.int8
    )


def long_image_space(num_channels: int, width: int, height: int) -> tspecs.Bounded:
    return scalar_box_space(0, 255, dtype=torch.long)
