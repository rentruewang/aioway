# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

import torch

from aioway.hop import Iter

from .thunks import Thunk

__all__ = ["UFunc", "TensorUFunc1"]


class UFunc(abc.ABC):
    """
    `UFunc`, inspired by `numpy`, stands for universal functions,
    and are the building blocks of other APIs (like thunks and iters).
    """

    @typing.no_type_check
    @abc.abstractmethod
    def __call__(self, *items) -> typing.Any:
        "A `UFunc` is callable (imperative)."

        raise NotImplementedError

    @typing.no_type_check
    @abc.abstractmethod
    def thunk(self, *thunks: Thunk) -> Thunk:
        "A `UFunc` takes thunks and tranform it into other thunks."

        raise NotImplementedError

    @typing.no_type_check
    @abc.abstractmethod
    def iter(self, *iters: Iter) -> Iter:
        "A `UFunc` takes iterators and tranform it into other iterators."

        raise NotImplementedError


class TensorUFunc1(UFunc, abc.ABC):
    @abc.abstractmethod
    def __call__(self, item: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def thunk(self, item: Thunk[torch.Tensor], /) -> Thunk[torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter(self, item: Iter[torch.Tensor], /) -> Iter[torch.Tensor]:
        raise NotImplementedError
