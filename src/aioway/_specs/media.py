# Copyright (c) AIoWay Authors - All Rights Reserved


import torch
from torchrl import data as rldata

from .gym import scalar_box_spec, unbounded_box_spec

__all__ = ["float_image", "byte_image", "long_image"]


def float_image(num_channels: int) -> rldata.Bounded:
    return scalar_box_spec(0, 1, shape=torch.Size([num_channels, -1, -1]))


def byte_image(num_channels: int) -> rldata.Unbounded:
    return unbounded_box_spec(
        shape=torch.Size([num_channels, -1, -1]), dtype=torch.int8
    )


def long_image(num_channels: int) -> rldata.Bounded:
    return scalar_box_spec(0, 255, dtype=torch.long)
