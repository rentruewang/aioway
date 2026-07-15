# Copyright (c) AIoWay Authors - All Rights Reserved

"The profile for `UFunc`."

import contextlib as ctxl
import dataclasses as dcls
import time
import typing
from collections import abc as cabc

from aioway._utils import AnyDict

if typing.TYPE_CHECKING:
    from .ufuncs import UFunc

__all__ = ["UFuncProf", "CallUFuncProf", "UFuncProfStack"]


@typing.runtime_checkable
class UFuncProf(typing.Protocol):
    """
    The profiler interface for `UFunc`.
    """

    def __call__(self, ufunc: UFunc, /) -> typing.ContextManager[typing.Any]:
        """
        The profiler exposes a callable.
        """


@dcls.dataclass
class _CallStat:
    count: int = 0
    """
    The number of calls to each `UFunc`.
    """

    elapsed: float = 0.0
    """
    The time that has passed for each `UFunc` calls.
    """


def _default_stat_dict() -> AnyDict[UFunc, _CallStat]:
    from .ufuncs import UFunc

    return AnyDict(UFunc)


@dcls.dataclass(frozen=True)
class CallUFuncProf(UFuncProf):
    """
    The `UFuncProf` that track calls to each `UFunc`s.
    """

    stats: AnyDict[UFunc, _CallStat] = dcls.field(default_factory=_default_stat_dict)

    @ctxl.contextmanager
    def __call__(self, ufunc: UFunc, /) -> cabc.Generator[None]:
        stat = self.stats[ufunc] if ufunc in self.stats else _CallStat()
        start = time.time()

        try:
            yield
        finally:
            end = time.time() - start

            stat.count += 1
            stat.elapsed += end

            self.stats[ufunc] = stat


@dcls.dataclass(frozen=True)
class UFuncProfStack(UFuncProf):
    """
    The stack, much like `contextlib`'s `ExitStack`, enters multiple profilers at once.
    """

    profs: cabc.Sequence[UFuncProf]
    "The stack of `UFuncProf`s."

    @ctxl.contextmanager
    def __call__(self, ufunc: UFunc, /) -> cabc.Generator[None]:
        "Enter all the `UFuncProf`s in the current stack."

        with ctxl.ExitStack() as stack:
            for prof in self.profs:
                stack.enter_context(prof(ufunc))

            yield
