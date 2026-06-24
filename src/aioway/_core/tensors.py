# Copyright (c) AIoWay Authors - All Rights Reserved

"UFunc for `torch.Tensor`s."

import dataclasses as dcls
from collections import abc as cabc

import torch

from .iters import Iter
from .ufuncs import LazyThunk, UFunc

__all__ = [
    "TensorUFunc1",
    "TensorUFunc1Thunk",
    "TensorUFunc1Iter",
    "TensorUFunc2",
    "TensorUFunc2Thunk",
    "TensorUFunc2Iter",
    "TensorUFuncN",
    "TensorUFuncNThunk",
    "TensorUFuncNIter",
]

type _TorchFunc1 = cabc.Callable[[torch.Tensor], torch.Tensor]
type _TorchFunc2 = cabc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
type _TorchFuncN = cabc.Callable[[cabc.Sequence[torch.Tensor]], torch.Tensor]


class TensorUFunc1(UFunc[torch.Tensor]):
    "The universal function type for `(torch.Tensor) -> torch.Tensor`."

    def __init__(self, func: _TorchFunc1, /) -> None:
        self._func = func

    def __call__(self, item: torch.Tensor, /) -> torch.Tensor:
        return self.func(item)

    def thunk(self, item: LazyThunk[torch.Tensor], /) -> LazyThunk[torch.Tensor]:
        return TensorUFunc1Thunk(self.func, item)

    def iter(self, item: Iter[torch.Tensor], /) -> Iter[torch.Tensor]:
        return TensorUFunc1Iter(self.func, item)

    @property
    def func(self):
        return self._func


@dcls.dataclass
class TensorUFunc1Thunk(LazyThunk[torch.Tensor]):
    "The thunk type for `(torch.Tensor) -> torch.Tensor`."

    func: _TorchFunc1
    input: LazyThunk[torch.Tensor]

    def __call__(self) -> torch.Tensor:
        return self.func(self.input())


@dcls.dataclass
class TensorUFunc1Iter(Iter[torch.Tensor]):
    "The iterator type for `(torch.Tensor) -> torch.Tensor`."

    func: _TorchFunc1
    input: Iter[torch.Tensor]

    def iterate(self) -> cabc.Iterator[torch.Tensor]:
        for batch in self.input:
            yield self.func(batch)


class TensorUFunc2(UFunc[torch.Tensor]):
    "The universal function type for `(torch.Tensor, torch.Tensor) -> torch.Tensor`."

    def __init__(self, func: _TorchFunc2, /) -> None:
        self._func = func

    def __call__(self, left: torch.Tensor, right: torch.Tensor, /) -> torch.Tensor:
        return self.func(left)

    def thunk(
        self, left: LazyThunk[torch.Tensor], right: LazyThunk[torch.Tensor], /
    ) -> LazyThunk[torch.Tensor]:
        return TensorUFunc2Thunk(self.func, left, right)

    def iter(
        self, left: Iter[torch.Tensor], right: Iter[torch.Tensor], /
    ) -> Iter[torch.Tensor]:
        return TensorUFunc2Iter(self.func, left, right)

    @property
    def func(self):
        return self._func


@dcls.dataclass
class TensorUFunc2Thunk(LazyThunk[torch.Tensor]):
    "The thunk type for `(torch.Tensor, torch.Tensor) -> torch.Tensor`."

    func: _TorchFunc2
    left: LazyThunk[torch.Tensor]
    right: LazyThunk[torch.Tensor]

    def __call__(self) -> torch.Tensor:
        return self.func(self.left(), self.right())


@dcls.dataclass
class TensorUFunc2Iter(Iter[torch.Tensor]):
    "The iterator type for `(torch.Tensor, torch.Tensor) -> torch.Tensor`."

    func: _TorchFunc2
    left: Iter[torch.Tensor]
    right: Iter[torch.Tensor]

    def __post_init__(self) -> None:
        if self.left.size != self.right.size:
            raise ValueError(
                "Cannot zip together 2 `Iter` of different size. "
                f"{self.left.size=}, {self.right.size=}."
            )

    def iterate(self) -> cabc.Iterator[torch.Tensor]:
        for batch_l, batch_r in zip(self.left, self.right):
            yield self.func(batch_l, batch_r)


class TensorUFuncN(UFunc[torch.Tensor]):
    "The universal function type for `(torch.Tensor) -> torch.Tensor`."

    def __init__(self, func: _TorchFuncN, /) -> None:
        self._func = func

    def __call__(self, item: cabc.Sequence[torch.Tensor], /) -> torch.Tensor:
        return self.func(item)

    def thunk(
        self, item: LazyThunk[cabc.Sequence[torch.Tensor]], /
    ) -> LazyThunk[torch.Tensor]:
        return TensorUFuncNThunk(self.func, item)

    def iter(self, item: Iter[cabc.Sequence[torch.Tensor]], /) -> Iter[torch.Tensor]:
        return TensorUFuncNIter(self.func, item)

    @property
    def func(self):
        return self._func


@dcls.dataclass
class TensorUFuncNThunk(LazyThunk[torch.Tensor]):
    "The thunk type for `(torch.Tensor) -> torch.Tensor`."

    func: _TorchFuncN
    inputs: LazyThunk[cabc.Sequence[torch.Tensor]]

    def __call__(self) -> torch.Tensor:
        return self.func(self.inputs())


@dcls.dataclass
class TensorUFuncNIter(Iter[torch.Tensor]):
    "The iterator type for `(torch.Tensor) -> torch.Tensor`."

    func: _TorchFuncN
    inputs: Iter[cabc.Sequence[torch.Tensor]]

    def iterate(self) -> cabc.Iterator[torch.Tensor]:
        for batch in self.inputs:
            yield self.func(batch)
