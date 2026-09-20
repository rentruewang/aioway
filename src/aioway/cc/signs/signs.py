# Copyright (c) AIoWay Authors - All Rights Reserved

"Registering `nn.Module`'s signatures, so we don't recompute them."

from torch import nn

from aioway._utils import Sign
from aioway.cc.regs import sign_reg

__all__ = ["nn_sign_skeleton", "register_nn_type_signature"]


def nn_sign_skeleton(module: type[nn.Module]) -> Sign:
    """
    Parse the `nn.Module.forward` without `self` or type.

    Caches the signature into `nn.Module` registry automatically.
    """

    sreg = sign_reg()

    if module not in sreg:
        register_nn_type_signature(module)

    return sreg[module]


def register_nn_type_signature(module: type[nn.Module]):
    """
    Register signature for the given `nn.Module` type.
    """

    if module in (reg := sign_reg()):
        return

    sign = Sign.from_nn_forward(module).strip_type()
    reg[module] = sign
