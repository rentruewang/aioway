# Copyright (c) AIoWay Authors - All Rights Reserved


import torch
from torchrl.data import tensor_specs as tspecs

from .gym import scalar_box_tspec, unbounded_box_tspec

__all__ = ["float_image_tspec", "byte_image_tspec", "long_image_tspec"]


def float_image_tspec(num_channels: int, width: int, height: int) -> tspecs.Bounded:
    return scalar_box_tspec(0, 1, shape=torch.Size([num_channels, width, height]))


def byte_image_tspec(num_channels: int, width: int, height: int) -> tspecs.Unbounded:
    return unbounded_box_tspec(
        shape=torch.Size([num_channels, width, height]), dtype=torch.int8
    )


def long_image_tspec(num_channels: int, width: int, height: int) -> tspecs.Bounded:
    return scalar_box_tspec(0, 255, dtype=torch.long)
