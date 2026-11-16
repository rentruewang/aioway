# Copyright (c) AIoWay Authors - All Rights Reserved

"Unroll `nn.Sequential` into `Op`s, so rules can see the layers inside."

from torch import nn

from aioway.ops import LayerOp, Op

__all__ = ["unroll_sequential"]


def unroll_sequential(seq: nn.Sequential, input: Op, /) -> Op:
    """
    The chain of `LayerOp`s that `seq` runs on `input`.

    Nested `nn.Sequential`s are unrolled too. Modules are shared with `seq`, not copied.
    """

    x = input

    for module in seq:
        if isinstance(module, nn.Sequential):
            x = unroll_sequential(module, x)
        else:
            x = LayerOp(module, x)

    return x
