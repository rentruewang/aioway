# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway.torch import (
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

__all__ = ["DoneThunk"]


@dcls.dataclass(frozen=True)
class DoneThunk[F: cabc.Callable]:
    """
    Stores the thunk's arguments, function, and output.
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

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    def upstream(self) -> cabc.Generator[torch.Tensor]:
        "Get the dependencies of the current thunk."

        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def downstream(self) -> cabc.Generator[torch.Tensor]:
        "Get the output of the current thunk."

        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is dcls.MISSING
