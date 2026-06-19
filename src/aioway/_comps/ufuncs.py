# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import torch

from .iters import Iter
from .thunks import Thunk

__all__ = ["UFunc", "TensorUFunc1"]


class UFunc(typing.Protocol):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

    __call__: cabc.Callable
    "A `UFunc` is callable (imperative)."

    thunk: cabc.Callable
    "A `UFunc` takes thunks and tranform it into other thunks."

    iter: cabc.Callable
    "A `UFunc` takes iterators and tranform it into other iterators."


class TensorUFunc1(UFunc, typing.Protocol):
    @abc.abstractmethod
    def __call__(self, item: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def thunk(self, item: Thunk[torch.Tensor], /) -> Thunk[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter(self, item: Iter[torch.Tensor], /) -> Iter[torch.Tensor]:
        raise NotImplementedError


class TensorUFunc2(UFunc, typing.Protocol):
    @abc.abstractmethod
    def __call__(self, left: torch.Tensor, right: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def thunk(
        self, left: Thunk[torch.Tensor], right: Thunk[torch.Tensor], /
    ) -> Thunk[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter(
        self, left: Iter[torch.Tensor], right: Iter[torch.Tensor], /
    ) -> Iter[torch.Tensor]:
        raise NotImplementedError
