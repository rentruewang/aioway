# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from .modules import NnInstr, nn_instr_dcls

__all__ = ["Dropout", "Dropout1d", "Dropout2d", "Dropout3d"]


@nn_instr_dcls
class _BaseDropout(NnInstr):
    NN: typing.ClassVar[type[nn.Module]] = NotImplemented

    p: float = 0.5
    "Probability of an element to be zeroed. Default: 0.5."

    inplace: bool = False
    "If set to True, will do this operation in-place. Default: `False`."


@nn_instr_dcls
class Dropout(_BaseDropout):
    """
    During training, randomly zeroes some of the elements
    of the input tensor with probability `p`.
    """

    NN = nn.Dropout


@nn_instr_dcls
class Dropout1d(_BaseDropout):
    """
    Randomly zero out entire channels (1D feature map).
    """

    NN = nn.Dropout1d


@nn_instr_dcls
class Dropout2d(_BaseDropout):
    """
    Randomly zero out entire channels (2D feature map).
    """

    NN = nn.Dropout2d


@nn_instr_dcls
class Dropout3d(_BaseDropout):
    """
    Randomly zero out entire channels (3D feature map).
    """

    NN = nn.Dropout3d
