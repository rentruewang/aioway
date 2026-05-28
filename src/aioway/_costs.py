# Copyright (c) AIoWay Authors - All Rights Reserved

"The costs management."

import contextlib as ctxl
import dataclasses as dcls
import functools
import operator
import typing
from collections import abc as cabc

from aioway._utils import Stack

__all__ = ["Cost", "CostSession", "current_session"]


_latest_session: CostSession | None = None
"The latest cost session."


@dcls.dataclass
class Cost:
    """
    The base unit for cost tracking.
    You should initialize and call `.commit()` to track it.
    """

    time: int
    "Time cost."

    memory: int
    "Memory cost."

    def __add__(self, other: Cost) -> typing.Self:
        return self.__elemwise(other, operator.add)

    def __sub__(self, other: Cost) -> typing.Self:
        return self.__elemwise(other, operator.sub)

    def __elemwise(
        self, other: Cost, ufunc: cabc.Callable[[int, int], int]
    ) -> typing.Self:
        return type(self)(
            time=ufunc(self.time, other.time),
            memory=ufunc(self.memory, other.memory),
        )

    def commit(self) -> None:
        cumsum = _cost_cumsum().top() + self
        _cost_cumsum().append(cumsum)

    @classmethod
    def zero(cls) -> typing.Self:
        return cls(time=0, memory=0)


class CostSession:
    """
    The cost session. Use the `.track()` function to track the costs in a new scope.
    Providing `.total()` function for summarization of costs.
    Not thread safe, but efficient in single threading context.
    """

    def __init__(self) -> None:
        self._before_count = len(_cost_cumsum())
        """
        These items, due to how scopes and stacks work (not thread safe),
        will not be modified in the scope of `self.track`.
        """

    def __len__(self) -> int:
        "The number of items, in the scope of this session."

        return len(_cost_cumsum()) - self._before_count

    def __getitem__(self, idx: int | slice, /) -> Cost:
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        if idx.step is not None:
            raise ValueError(
                "Only support contiguous cost views right now. "
                "Do not specify the step for slices."
            )

        return self.__getitem_slice(idx.start, idx.stop)

    def __getitem_slice(self, start: int | None, end: int | None):
        start = start if start is not None else 0
        end = end if end is not None else len(self)

        end_cost = _cost_cumsum()[end + self._before_count - 1]
        start_cost = _cost_cumsum()[start + self._before_count - 1]
        return end_cost - start_cost

    def total(self) -> Cost:
        return self[:]

    @ctxl.contextmanager
    def track(self) -> cabc.Generator[typing.Self]:
        """
        Track the costs in the `CostSession`.
        """

        global _latest_session

        try:
            with _set_latest_session(self):
                yield self
        finally:
            # When the scope exits, clean up all the costs (in the current session).
            assert len(self) >= 0
            _cost_cumsum().truncate(self._before_count)


@ctxl.contextmanager
def _set_latest_session(session: CostSession):

    global _latest_session
    before = _latest_session
    _latest_session = session

    try:
        yield
    finally:
        _latest_session = before


@functools.lru_cache(maxsize=1)
def _cost_cumsum() -> Stack[Cost]:
    """
    The cumsum of costs. Since we are doing a lot of slice summation (and no setitem),
    this gives O(1) slice summation at the cost of item access being slower.

    The stack itself always has a minimum size of 1 (concetual 0).
    """

    return Stack([Cost.zero()])


def current_session() -> CostSession:
    "Get the currently active session."

    if _latest_session is None:
        raise RuntimeError(
            "You have not started tracking cost with `CostSession().track()` yet."
        )

    return _latest_session
