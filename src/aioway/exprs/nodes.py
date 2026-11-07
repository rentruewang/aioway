# Copyright (c) AIoWay Authors - All Rights Reserved


import abc
import typing
from collections import abc as cabc

from .exprs import Expr, expr_dcls

__all__ = ["CallExpr"]


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
    def __call__(self):
        raise NotImplementedError

    @typing.final
    def inputs(self):
        raise NotImplementedError
