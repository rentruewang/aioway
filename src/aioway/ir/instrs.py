# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import abc
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch
from torch import nn
from torch.utils import _pytree as pytree

from aioway.t import (
    TList,
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

__all__ = ["Instr", "FCall", "MCall"]


@typing.dataclass_transform(frozen_default=True, eq_default=False)
def instr_dcls[T](cls):
    """
    The dataclass transform wrapper for `Instr` classes.

    This also registeres the class to `pytree`.
    """

    cls = dcls.dataclass(frozen=True, eq=False, repr=False)(cls)
    pytree.register_dataclass(cls)
    return cls


# The base instruction class ====


@instr_dcls
class Instr(abc.ABC):
    "The instruction base class."

    def tree_map(self, func: cabc.Callable):
        """
        Call the `tree_map` function on the `Instr`.
        """

        return pytree.tree_map(func=func, tree=self)

    def tree_map_only(self, types: type | tuple[type, ...], func: cabc.Callable):
        """
        Call the `tree_map_only` function on the `Instr`.
        """

        return pytree.tree_map_only(types, func=func, tree=self)

    @functools.cached_property
    def inputs(self) -> TList:
        return self._inputs()

    @functools.cached_property
    def outputs(self) -> TList:
        return self._outputs()

    @abc.abstractmethod
    def _inputs(self) -> TList:
        "The list of tensors in the inputs."

        raise NotImplementedError

    @abc.abstractmethod
    def _outputs(self) -> TList:
        "The list of tensors in the outputs."

        raise NotImplementedError


# The implementations. Right now only `FCall` exists. ====


@instr_dcls
class FCall[F: cabc.Callable](Instr):
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
        return TList(self._in_tensors())

    @typing.override
    def _outputs(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList(self._out_tensors())

    def _in_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def _out_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is not dcls.MISSING


@instr_dcls
class MCall(FCall):
    """
    `MCall` is the module version of `FCall` instruction.
    """

    _: dcls.KW_ONLY

    parents: tuple[nn.Module, ...]
    "The parent modules that calls this current thunk. It's a stack."

    def __repr__(self) -> str:
        return super().__repr__() + f" [{len(self.parents)} parents]"
