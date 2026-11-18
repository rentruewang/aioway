# Copyright (c) AIoWay Authors - All Rights Reserved

"Unroll `nn.Sequential` into `Op`s, so rules can see the layers inside."

import torch
from torch import nn

from aioway._signs import Param
from aioway.ops import FuncInputOp, LayerOp, Op, OpFunc
from aioway.t import TensorDomain

__all__ = ["unroll_sequential"]


def unroll_sequential(seq: nn.Sequential, /) -> OpFunc:
    """
    The `OpFunc` of `seq`: a chain of `LayerOp`s on one input, `input`.

    Nested `nn.Sequential`s are unrolled too. Modules are shared with `seq`, not copied.
    """

    input = FuncInputOp(param=Param.build("input"), domain=TensorDomain())
    return OpFunc(_unroll(seq, input))


def _unroll(seq: nn.Sequential, x: Op) -> Op:
    for module in seq:
        if isinstance(module, nn.Sequential):
            x = _unroll(module, x)
        else:
            x = LayerOp(module, x)

    return x
