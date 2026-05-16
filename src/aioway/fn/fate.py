# Copyright (c) AIoWay Authors - All Rights Reserved

"`FateFn` is an adaptor for `aioway.fate`."

import dataclasses as dcls
import typing

from aioway.fate import Fate, find_fate

from .fn import Fn
from .tensors import TDispatchFn

__all__ = ["FateFn"]


@typing.final
@dcls.dataclass(frozen=True)
class FateFn(Fn):
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

    @typing.override
    def do(self) -> object:
        return self.fate.do()

    def inputs(self):
        yield from self.fate.inputs()

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
        fate = find_fate(thunk.func, *thunk.args, **thunk.kwargs)

        if fate is NotImplemented:
            return NotImplemented

        else:
            return cls(fate=fate, original=thunk)
