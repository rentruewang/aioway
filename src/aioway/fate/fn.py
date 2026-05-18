# Copyright (c) AIoWay Authors - All Rights Reserved

"An adaptor of `aioway.op`, brining in `Fate` / `Might` to `Fn`."

import dataclasses as dcls
import typing

from aioway._common._torch import (
    is_aten_op,
)
from aioway.fake import enabled_fake_mode
from aioway.fn import TorDisFn

from .fate import Fate, find_fate

__all__ = ["FateFn"]


@typing.final
@dcls.dataclass(frozen=True)
class FateFn:
    """
    `FateFn` wraps a `Fate` object, which is split out so as to declutter subclasses for `Fn`.

    Each `Fate` is an implementation of an IR, and each IR can have multiple `Fate`s,
    each handling a subset of parameters (if `Fate.ok` is `False`, it's discarded.)
    """

    fate: Fate
    """
    The `Fate` object that ends up being selected.
    """

    original: TorDisFn
    "The original `TorDisFn` from which the `Fate` is translated."

    def __repr__(self) -> str:
        return repr(self.fate)

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
    def find_fate(cls, thunk: TorDisFn) -> typing.Self:
        if not enabled_fake_mode():
            return NotImplemented

        # For now, `Fate` supports aten, because `torchvision`, `torchcodec` rely on real data,
        # they do not have a good `Fate` to implement for now.
        # In those operations, real mode is force enabled right now.
        # See aioway#204 issue.
        if not is_aten_op(thunk.func):
            return NotImplemented

        fate = find_fate(thunk.func, *thunk.args, **thunk.kwargs)

        if fate is NotImplemented:
            return NotImplemented

        else:
            return cls(fate=fate, original=thunk)
