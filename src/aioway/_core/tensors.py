# Copyright (c) AIoWay Authors - All Rights Reserved

"UFunc for `torch.Tensor`s."

import abc
import typing

import torch

from .iters import Iter
from .thunks import Thunk
from .ufuncs import UFunc

__all__ = ["TensorUFunc1"]


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
