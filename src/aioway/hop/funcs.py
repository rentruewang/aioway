# Copyright (c) AIoWay Authors - All Rights Reserved

"The high level operators that are just torch functions / data."

import abc
import typing
from collections import abc as cabc

import torch

from .hop import Hop, TensorHop, hop_dcls

__all__ = ["TensorHop", "CatHop", "StackHop"]


@hop_dcls
class _CatStackHop(TensorHop, abc.ABC):
    """
    The `Hop` implementation for `torch.cat` / `torch.stack`.
    """

    FUNCTION: typing.ClassVar[cabc.Callable[..., torch.Tensor]]
    """
    Either `torch.cat` or `torch.stack` or something like those.
    It's typed as `cabc.Callable[..., torch.Tensor]`
    because `torch` has complicated type stubs, annoying to deal with.
    """

    tensors: list[Hop]
    "The list of `Hop` that would evaluate each to a `torch.Tensor`."

    dim: int = 0
    "The `dim` flag that would be passed to `.function`."

    @typing.override
    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for tensors in zip(*self.tensors):
            yield self.FUNCTION(list(tensors), dim=self.dim)


@hop_dcls
class CatHop(_CatStackHop):
    "The `Hop` backed by `torch.cat`."

    FUNCTION = torch.cat


@hop_dcls
class StackHop(_CatStackHop):
    "The `Hop` backed by `torch.stack`."

    FUNCTION = torch.stack
