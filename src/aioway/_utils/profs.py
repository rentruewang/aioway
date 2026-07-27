# Copyright (c) AIoWay Authors - All Rights Reserved

"The profilers."

import abc
import contextlib as ctxl
import dataclasses as dcls
import functools
import time
import typing
from collections import abc as cabc

from aioway._utils import AnyDict

from .types import Stack

__all__ = ["Profiler", "ProfilerStack", "CallerProfiler", "ProfilerCollection"]


class Profiler(abc.ABC):
    """
    The profiler interface.
    """

    @abc.abstractmethod
    def __call__(self, item: typing.Any, /) -> typing.ContextManager[typing.Self]:
        "The profiler exposes a callable that can be used to enter."
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class ProfilerStack:
    "The stack of profilers. Each profiler has its own stack."

    stack_dict: AnyDict[Profiler, Stack[typing.Any]]
    "The backing stacks."

    @ctxl.contextmanager
    def track(self, profiler: Profiler, identifier: typing.Any) -> cabc.Generator[None]:
        "Track the scope marked by `profiler` and `identifier`."

        self._setup_append(profiler, identifier)
        try:
            yield
        finally:
            self._pop_teardown(profiler)

    def _setup_append(self, profiler: Profiler, identifier: typing.Any):
        "Add the identifier to the stack, if not exists, add to `self.stacks`."

        if profiler not in self.stack_dict:
            self.stack_dict[profiler] = Stack()

        self.stack_dict[profiler].append(identifier)

    def _pop_teardown(self, profiler: Profiler):
        "Remove the identifier from the stack, if stack empty, remove from `self.stacks`."

        self.stack_dict[profiler].pop()

        if not self.stack_dict[profiler]:
            del self.stack_dict[profiler]

    @classmethod
    @functools.cache
    def singleton(cls) -> typing.Self:
        """
        The global profiler stack, cached.
        """

        stack_dict = AnyDict(Profiler)
        return cls(stack_dict)


@dcls.dataclass
class _CallStat:
    count: int = 0
    """
    The number of calls.
    """

    elapsed: float = 0.0
    """
    The time that has passed for each call.
    """


@dcls.dataclass(frozen=True)
class CallerProfiler(Profiler):
    """
    The `Profiler` that track calls.
    """

    stats: AnyDict[typing.Any, _CallStat] = dcls.field(default_factory=AnyDict)

    @typing.override
    @ctxl.contextmanager
    def __call__(self, item: typing.Any, /) -> cabc.Generator[typing.Self]:
        stat = self.stats[item] if item in self.stats else _CallStat()
        start = time.time()

        try:
            yield self
        finally:
            end = time.time() - start

            stat.count += 1
            stat.elapsed += end

            self.stats[item] = stat


@dcls.dataclass(frozen=True)
class ProfilerCollection(Profiler):
    """
    The stack, much like `contextlib`'s `ExitStack`, enters multiple profilers at once.
    """

    profs: cabc.Sequence[Profiler]
    "The sequence of `Profiler`s."

    @ctxl.contextmanager
    def __call__(self, item: typing.Any, /) -> cabc.Generator[typing.Self]:
        "Enter all the `Profiler`s in the current stack."

        with ctxl.ExitStack() as stack:
            for prof in self.profs:
                stack.enter_context(prof(item))

            yield self
