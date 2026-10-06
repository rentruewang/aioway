# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway.t import (
    TList,
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

from .instrs import Instr, instr_dcls

__all__ = ["FuncCall", "ModuleCall"]


@instr_dcls
class FuncCall[F: cabc.Callable](Instr):
    """
    An instruction representing a function call.
    """

    _: dcls.KW_ONLY

    func: F
    "The callable that the thunk calls."

    args: tuple[typing.Any, ...]
    "The arguments fed to the function."

    kwargs: dict[str, typing.Any]
    "The keyword arguments fed to the function."

    result: typing.Any = dcls.MISSING
    "The result. If `dcls.MISSING`, the thunk is not called yet."

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

    def __eq__(self, other) -> bool:
        """
        `FCall` will not implement `__eq__`.
        """

        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        def maybe_name(func):
            try:
                return func.__name__
            except AttributeError:
                return repr(func)

        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(maybe_name(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @typing.override
    def _inputs(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList.from_iterable(self._in_tensors())

    @typing.override
    def _outputs(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList.from_iterable(self._out_tensors())

    def _in_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def _out_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is not dcls.MISSING


@instr_dcls
class ModuleCall(FuncCall):
    """
    The `nn.Module` version of `FCall` instruction.
    Handles the super classes.
    """

    _: dcls.KW_ONLY

    parents: tuple[nn.Module, ...]
    "The parent modules that calls this current thunk. It's a stack."

    def __repr__(self) -> str:
        return super().__repr__() + f" [{len(self.parents)} parents]"
