# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Hop`s backed by torch functions."

import typing
from collections import abc as cabc

import torch

from aioway._utils import is_list_of

from .hop import Hop, hop_dcls

__all__ = ["CatHop"]


@hop_dcls
class _CatStackFunc(Hop):
    FUNCTION: cabc.Callable[..., torch.Tensor]
    """
    Either `torch.cat` or `torch.stack` or something like those.
    It's typed as `cabc.Callable[..., torch.Tensor]`
    because `torch` has complicated type stubs, annoying to deal with.
    """

    dim: int = 0
    "The `dim` flag that would be passed to `.function`."

    inputs: list[Hop]
    "The list of `Hop` that would evaluate each to a `torch.Tensor`."

    @typing.override
    def do(self) -> torch.Tensor:
        tensors = [i.do() for i in self.inputs]

        if not is_list_of(torch.Tensor)(tensors):
            raise TypeError(f"Expected a `list[torch.Tensor]`, but {tensors=}.")

        return self.FUNCTION(tensors, dim=self.dim)

    @typing.override
    def deps(self) -> cabc.Iterator[Hop]:
        yield from self.inputs


@hop_dcls
class CatHop(_CatStackFunc):
    "The `Hop` backed by `torch.cat`."

    FUNCTION = torch.cat


@hop_dcls
class StackHop(_CatStackFunc):
    "The `Hop` backed by `torch.stack`."

    FUNCTION = torch.stack
