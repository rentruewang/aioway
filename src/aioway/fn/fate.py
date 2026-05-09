# Copyright (c) AIoWay Authors - All Rights Reserved

"Integration package between `aioway.fate` and `aioway.fn`."

import dataclasses as dcls
import typing
from collections import abc as cabc

import torch

from aioway._common import HasParam, find_nested_tensors
from aioway.fate import Fate, find_fate

from .modes import TDispatchFn

__all__ = ["FateFn"]


@typing.final
@dcls.dataclass(frozen=True)
class FateFn(HasParam):
    """
    `FateFn` wraps a `Fate` object, which is split out so as to declutter subclasses for `Fn`.

    Each `Fate` is an implementation of an IR, and each IR can have multiple `Fate`s,
    each handling a subset of parameters (if `Fate.ok` is `False`, it's discarded.)
    """

    fate: Fate
    """
    The `Fate` object that ends up being selected.
    """

    original: TDispatchFn
    "The original `TorchDispatchFn` from which the `Fate` is translated."

    def __repr__(self) -> str:
        return repr(self.fate)

    def do(self) -> torch.Tensor:
        return self.fate.do()

    @typing.override
    def tensors(self) -> cabc.Iterator[torch.Tensor]:
        yield from find_nested_tensors(self.fate)

    @property
    def func(self):
        return self.original.func

    @property
    def args(self):
        return self.original.args

    @property
    def kwargs(self):
        return self.original.kwargs

    @classmethod
    def find_fate(cls, thunk: TDispatchFn) -> typing.Self:
        if (
            fate := find_fate(thunk.func, *thunk.args, **thunk.kwargs)
        ) is NotImplemented:
            return NotImplemented

        return cls(fate=fate, original=thunk)
