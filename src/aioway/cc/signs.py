# Copyright (c) AIoWay Authors - All Rights Reserved

"Registering `nn.Module`'s signatures, so we don't recompute them."

from torch import nn

from aioway._utils import Sign

from .regs import sign_reg

__all__ = ["nn_sign_skeleton", "sign_reg"]


def nn_sign_skeleton(module: type[nn.Module]) -> Sign:
    """
    Parse the `nn.Module.forward` without `self` or type.

    Caches the signature into `nn.Module` registry automatically.
    """

    sreg = sign_reg()

    if module not in sreg:
        sign = Sign.from_nn_forward(module).strip_type()
        sreg[module] = sign

    return sreg[module]
