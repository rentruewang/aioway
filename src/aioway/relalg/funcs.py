# Copyright (c) AIoWay Authors - All Rights Reserved

"The high level operators that are just torch functions / data."

import abc
import typing
from collections import abc as cabc

import torch

from aioway._core import Iter, TensorIter, node_dcls

__all__ = ["CatIter", "StackIter"]


@node_dcls
class _CatStackIter(TensorIter, abc.ABC):
    """
    The `Iter` implementation for `torch.cat` / `torch.stack`.
    """

    FUNCTION: typing.ClassVar[cabc.Callable[..., torch.Tensor]]
    """
    Either `torch.cat` or `torch.stack` or something like those.
    It's typed as `cabc.Callable[..., torch.Tensor]`
    because `torch` has complicated type stubs, annoying to deal with.
    """

    tensors: list[Iter[torch.Tensor]]
    "The list of `Iter` that would evaluate each to a `torch.Tensor`."

    dim: int = 0
    "The `dim` flag that would be passed to `.function`."

    @typing.override
    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for tensors in zip(*self.tensors):
            yield self.FUNCTION(list(tensors), dim=self.dim)


@node_dcls
class CatIter(_CatStackIter):
    "The `Iter` backed by `torch.cat`."

    FUNCTION = torch.cat


@node_dcls
class StackIter(_CatStackIter):
    "The `Iter` backed by `torch.stack`."

    FUNCTION = torch.stack
