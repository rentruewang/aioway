# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Hop`s backed by torch functions."

import typing
from collections import abc as cabc

import torch

from aioway._fn import thunk_dcls
from aioway._utils import is_list_of

from .hop import HopFwd, HopInit, hop_init_dcls

__all__ = ["CatHop", "StackHop"]


@thunk_dcls
class FuncHopFwd(HopFwd):
    """
    The `HopFwd` implementation for functions.
    """

    func: cabc.Callable[..., typing.Any]
    args: tuple[typing.Any, ...]
    kwargs: dict[str, typing.Any]

    def forward(self) -> object:
        return self.func(*self.args, **self.kwargs)


@hop_init_dcls
class _CatStackHopInit(HopInit):
    FUNCTION: cabc.Callable[..., torch.Tensor]
    """
    Either `torch.cat` or `torch.stack` or something like those.
    It's typed as `cabc.Callable[..., torch.Tensor]`
    because `torch` has complicated type stubs, annoying to deal with.
    """

    dim: int = 0
    "The `dim` flag that would be passed to `.function`."

    inputs: list[HopInit]
    "The list of `Hop` that would evaluate each to a `torch.Tensor`."

    @typing.override
    def __call__(self):
        tensors = [i() for i in self.inputs]

        if not is_list_of(torch.Tensor)(tensors):
            raise TypeError(f"Expected a `list[torch.Tensor]`, but {tensors=}.")

        return FuncHopFwd(func=self.FUNCTION, args=(tensors,), kwargs={"dim": self.dim})


@hop_init_dcls
class CatHop(_CatStackHopInit):
    "The `Hop` backed by `torch.cat`."

    FUNCTION = torch.cat


@hop_init_dcls
class StackHop(_CatStackHopInit):
    "The `Hop` backed by `torch.stack`."

    FUNCTION = torch.stack
