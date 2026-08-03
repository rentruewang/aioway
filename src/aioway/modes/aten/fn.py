# Copyright (c) AIoWay Authors - All Rights Reserved

"An adaptor of `Aten` and `Thunk`, using `Aten` in fake modes."

import dataclasses as dcls
import logging
import typing

from aioway._torch import is_aten_op

from .aten import Aten, find_aten

if typing.TYPE_CHECKING:
    from aioway.modes import TorchDispThunk

__all__ = ["AtenThunk"]

LOGGER = logging.getLogger(__name__)


@typing.final
@dcls.dataclass(frozen=True)
class AtenThunk:
    """
    `AtenThunk` wraps a `Aten` object, which is split out so as to declutter subclasses for `Thunk`.

    Each `Aten` is an implementation of an IR, and each IR can have multiple `Aten`s,
    each handling a subset of parameters (if `Aten.ok` is `False`, it's discarded.)
    """

    aten: Aten
    """
    The `Aten` object that ends up being selected.
    """

    original: TorchDispThunk
    "The original `TorchDispThunk` from which the `Aten` is translated."

    def __repr__(self) -> str:
        return repr(self.aten)

    def __call__(self):
        return self.aten()

    def inputs(self):
        yield from self.aten.inputs()

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
    def from_thunk(cls, thunk: TorchDispThunk) -> typing.Self | None:
        LOGGER.debug("Resolving `Aten` object for %s", thunk)

        # For now, `Aten` only supports aten,
        # because `torchvision`, `torchcodec` rely on real data,
        # they do not have a good `Aten` to implement for now.
        # In those operations, real mode is force enabled right now.
        # See aioway#204 issue.
        if not is_aten_op(thunk.func):
            LOGGER.debug("%s is not aten.", thunk)
            return None

        aten = find_aten(thunk)

        if aten is None:
            LOGGER.debug("`Aten` for %s not found.", thunk)
            return None

        else:
            LOGGER.debug("`Aten` for %s found: %s.", thunk, aten)
            return cls(aten=aten, original=thunk)
