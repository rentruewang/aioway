# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway._utils import Sign
from aioway.torch import (
    find_nested_tensors,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

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
    ) -> None:
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._result = result

        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

        # Try binding, if not this would fail with `TypeError`.
        _ = self._signature.bind(*self.args, **self.kwargs)

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
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def result(self):
        return self._result

    @functools.cached_property
    def _signature(self) -> Sign:
        return Sign.from_callable(self._func)


@dcls.dataclass
class TorchDagNode[C: cabc.Callable]:
    function: C
    inputs: list[int]
    outputs: list[int]


class TorchDag:
    def __init__(self, execs):
        pass
