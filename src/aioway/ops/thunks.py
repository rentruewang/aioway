# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

from torch import nn

from aioway._signs import Sign

from .ops import Op, exec_op

__all__ = ["CallOp", "ModuleThunkOp"]


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

    @property
    def sign(self) -> Sign:
        return Sign.from_callable(self.func)


class ModuleOp(Op, abc.ABC):
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
