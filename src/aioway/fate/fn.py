# Copyright (c) AIoWay Authors - All Rights Reserved

"An adaptor of `Fate` and `Fn`, using `Fate` in fake modes."

import dataclasses as dcls
import logging
import typing

from aioway._fn import Fn
from aioway._torch import is_aten_op

from .fate import Fate, find_fate

if typing.TYPE_CHECKING:
    from aioway.modes import TorchDispFn

__all__ = ["FateFn"]

LOGGER = logging.getLogger(__name__)


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

    original: TorchDispFn
    "The original `TorchDispFn` from which the `Fate` is translated."

    def __repr__(self) -> str:
        return repr(self.fate)

    def __call__(self):
        return self.fate()

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
    def find_fate(cls, thunk: TorchDispFn) -> typing.Self | None:
        LOGGER.debug("Resolving `Fate` object for %s", thunk)

        # For now, `Fate` only supports aten,
        # because `torchvision`, `torchcodec` rely on real data,
        # they do not have a good `Fate` to implement for now.
        # In those operations, real mode is force enabled right now.
        # See aioway#204 issue.
        if not is_aten_op(thunk.func):
            LOGGER.debug("%s is not aten.", thunk)
            return None

        fate = find_fate(thunk)

        if fate is None:
            LOGGER.debug("Fate for %s not found.", thunk)
            return None

        else:
            LOGGER.debug("Fate for %s found: %s.", thunk, fate)
            return cls(fate=fate, original=thunk)
