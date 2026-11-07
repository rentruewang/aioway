# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch import nn

from aioway._pytrees import tree_leaves
from aioway._signs import Sign

from .exprs import Expr, expr_dcls

__all__ = ["CallExpr", "ModuleExpr"]


@expr_dcls
class CallExpr[F: cabc.Callable](Expr, abc.ABC):
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
    def evaluate(self):
        return self.func(*self.args, **self.kwargs)

    @typing.override
    def deps(self):
        for expr in tree_leaves(
            [self.args, self.kwargs], is_leaf=lambda e: isinstance(e, Expr)
        ):
            if isinstance(expr, Expr):
                yield expr

    @property
    def sign(self) -> Sign:
        return Sign.from_callable(self.func)


@expr_dcls
class ModuleExpr(CallExpr[nn.Module]):
    """
    The `nn.Module` version of `CallExpr`.
    """

    func: nn.Module
    "The `nn.Module` held by the expression."

    @property
    def sign(self) -> Sign:
        return Sign.from_nn_forward(self.func)
