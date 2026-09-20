# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.torch import find_nested_tensors
import typing
from collections import abc as cabc

import torch

from aioway._utils import decomp_flatten
from aioway.torch import render_tensor_func_short, replace_tensors_with_attr

__all__ = ["DoneTorchThunk"]


class DoneTorchThunk[F: cabc.Callable]:
    "Stores the thunk and output."

    def __init__(
        self,
        *,
        func: cabc.Callable,
        args: tuple,
        kwargs: dict[str, typing.Any],
        result: typing.Any,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = result

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
