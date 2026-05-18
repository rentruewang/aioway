# Copyright (c) AIoWay Authors - All Rights Reserved

"An adaptor of `aioway.op`, brining in `Fate` / `Might` to `Fn`."

import dataclasses as dcls
import typing

from .modes import NnInitFn, TorDisFn

if typing.TYPE_CHECKING:
    from aioway.fate import Fate
    from aioway.might import Might

__all__ = ["FateFn", "MightFn"]


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
        from aioway.fate import find_fate

        fate = find_fate(thunk.func, *thunk.args, **thunk.kwargs)

        if fate is NotImplemented:
            return NotImplemented

        else:
            return cls(fate=fate, original=thunk)


@dcls.dataclass
class MightFn:
    """
    `MightFn` are `Fn` that wrap `Might`s, which are supported `nn.Module` ops.
    """

    might: Might
    "The `Might` instance."

    def do(self) -> object:
        return self.might.do()

    @classmethod
    def find_might(cls, thunk: NnInitFn) -> typing.Self:
        from aioway.might import find_might

        might = find_might(thunk.func, *thunk.args, **thunk.kwargs)

        if might is NotImplemented:
            return NotImplemented

        else:
            return cls(might)
