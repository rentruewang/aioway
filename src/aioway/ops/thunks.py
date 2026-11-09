# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from torch import nn

from aioway._pytrees import tree_leaves
from aioway._signs import Sign

from .ops import Op, exec_op, op_dcls

__all__ = ["CallOp", "ModuleOp"]


@op_dcls
class CallOp[F: cabc.Callable](Op):
    """
    The base class denoting a function call.
    """

    func: F
    """
    The function that the module stores.
    """

    args: tuple[typing.Any, ...]
    """
    The arguments tuple.
    """

    kwargs: dict[str, typing.Any]
    """
    The mapping storing kwargs.
    """

    @typing.override
    def execute(self):
        args = [exec_op(arg) for arg in self.args]
        kwargs = {key: exec_op(val) for key, val in self.kwargs.items()}

        return self.func(*args, **kwargs)

    @typing.override
    def deps(self):
        for expr in tree_leaves(
            [self.args, self.kwargs], is_leaf=lambda e: isinstance(e, Op)
        ):
            if isinstance(expr, Op):
                yield expr

    @property
    def sign(self) -> Sign:
        return Sign.from_callable(self.func)


@op_dcls
class ModuleOp(CallOp[nn.Module]):
    """
    The `nn.Module` version of `CallOp`.
    """

    func: nn.Module
    "The `nn.Module` held by the expression."

    @property
    def sign(self) -> Sign:
        return Sign.from_nn_forward(self.func)
