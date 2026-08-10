# Copyright (c) AIoWay Authors - All Rights Reserved


import torch
from torchrl import data as rldata

from .gym import scalar_box_spec, unbounded_box_spec

__all__ = ["float_image_spec", "byte_image_spec", "long_image_spec"]


def float_image_spec(num_channels: int, width: int, height: int) -> rldata.Bounded:
    return scalar_box_spec(0, 1, shape=torch.Size([num_channels, width, height]))


def byte_image_spec(num_channels: int, width: int, height: int) -> rldata.Unbounded:
    return unbounded_box_spec(
        shape=torch.Size([num_channels, width, height]), dtype=torch.int8
    )


def long_image_spec(num_channels: int, width: int, height: int) -> rldata.Bounded:
    return scalar_box_spec(0, 255, dtype=torch.long)
