# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from .init import MessInit

__all__ = ["Dropout", "Dropout1d", "Dropout2d", "Dropout3d"]


class _BaseDropout(MessInit):
    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    p: float = 0.5
    "Probability of an element to be zeroed. Default: 0.5."

    inplace: bool = False
    "If set to True, will do this operation in-place. Default: `False`."


class Dropout(_BaseDropout):
    """
    During training, randomly zeroes some of the elements
    of the input tensor with probability `p`.
    """

    KEY = nn.Dropout


class Dropout1d(_BaseDropout):
    """
    Randomly zero out entire channels (1D feature map).
    """

    KEY = nn.Dropout1d


class Dropout2d(_BaseDropout):
    """
    Randomly zero out entire channels (2D feature map).
    """

    KEY = nn.Dropout2d


class Dropout3d(_BaseDropout):
    """
    Randomly zero out entire channels (3D feature map).
    """

    KEY = nn.Dropout3d
