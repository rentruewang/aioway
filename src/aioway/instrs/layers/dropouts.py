# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn
from torchrl.data import tensor_specs as tspecs

from ..nn import NnInstr, instr_dcls

__all__ = ["Dropout", "Dropout1d", "Dropout2d", "Dropout3d"]


@instr_dcls
class _BaseDropout(NnInstr):
    p: float = 0.5
    "Probability of an element to be zeroed. Default: 0.5."

    inplace: bool = False
    "If set to True, will do this operation in-place. Default: `False`."


@instr_dcls
class Dropout(_BaseDropout):
    """
    During training, randomly zeroes some of the elements
    of the input tensor with probability `p`.
    """

    NN = nn.Dropout


@instr_dcls
class Dropout1d(_BaseDropout):
    """
    Randomly zero out entire channels (1D feature map).
    """

    NN = nn.Dropout1d


@instr_dcls
class Dropout2d(_BaseDropout):
    """
    Randomly zero out entire channels (2D feature map).
    """

    NN = nn.Dropout2d


@instr_dcls
class Dropout3d(_BaseDropout):
    """
    Randomly zero out entire channels (3D feature map).
    """

    NN = nn.Dropout3d


@Dropout.deductor().register
@Dropout1d.deductor().register
@Dropout2d.deductor().register
@Dropout3d.deductor().register
def norm_deduct(self, input: tspecs.Unbounded):
    return input
