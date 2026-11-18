# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch import nn

from aioway._signs import Sign
from aioway.t import AnyDomain, Domain

from ._utils import exec_nested_ops
from .ops import Op

__all__ = ["CallOp", "ModuleThunkOp", "ModuleOp"]


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

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return [AnyDomain() for _ in self.__deps__]

    @typing.override
    def _infer_domain(self) -> Domain:
        return AnyDomain()

    @typing.override
    def execute(self):
        args = exec_nested_ops(self.args)
        kwargs = exec_nested_ops(self.kwargs)

        return self.func(*args, **kwargs)

    @property
    def sign(self) -> Sign:
        return Sign.from_callable(self.func)


class ModuleOp(Op[typing.Any], abc.ABC):
    """
    A marker that the `Op` is an `nn.Module`.
    """

    @property
    @abc.abstractmethod
    def nn(self) -> nn.Module:
        """
        The underlying `nn.Module`.
        """
        raise NotImplementedError


class ModuleThunkOp(CallOp[nn.Module], ModuleOp):
    """
    The `nn.Module` version of `CallOp`.
    """

    func: nn.Module
    "The `nn.Module` held by the expression."

    @property
    def sign(self) -> Sign:
        return Sign.from_nn_forward(self.func)

    @property
    def nn(self) -> nn.Module:
        return self.func
