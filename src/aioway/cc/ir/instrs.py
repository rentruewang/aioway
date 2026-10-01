# Copyright (c) AIoWay Authors - All Rights Reserved

"The instructions themselves."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway.t import (
    TList,
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

__all__ = ["FCall"]


@dcls.dataclass(frozen=True, eq=False, repr=False)
class FCall[F: cabc.Callable]:
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
        `ThunkNode` will not implement `__eq__`.
        """
        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @functools.cached_property
    def inputs(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList(self._in_tensors())

    @functools.cached_property
    def outputs(self) -> TList:
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
