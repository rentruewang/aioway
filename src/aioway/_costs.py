# Copyright (c) AIoWay Authors - All Rights Reserved

"The costs management."

import contextlib as ctxl
import dataclasses as dcls
import operator
import typing
from collections import abc as cabc

from aioway._utils import Stack

__all__ = ["Cost", "CostSession", "track_cost", "current_session"]


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
        """
        Record the cost to the session. No-op when no session active.
        """

        if (sess := current_session()) is None:
            return

        # Using this rather than the `_COST_CUMSUM` variable directly.
        # This forces `current_session()` to be `not None`,
        # so we always clean it up via scopes (don't append infinitely).
        sess.record(self)

    @classmethod
    def zero(cls) -> typing.Self:
        return cls(time=0, memory=0)


class CostSession:
    """
    The cost session. Use the `track_cost()` function to track the costs in a new scope.
    Providing `.sum()` function for summarization of costs.
    Not thread safe, but efficient in single threading context.
    """

    _COST_CUMSUM: typing.ClassVar[Stack[Cost]] = Stack([Cost.zero()])
    """
    The cumsum of costs. Since we are doing a lot of slice summation (and no setitem),
    this gives O(1) slice summation at the cost of item access being slower.

    The stack itself always has a minimum size of 1 (concetual 0).
    """

    def __init__(self) -> None:
        self._before_count = len(self._COST_CUMSUM)
        """
        These items, due to how scopes and stacks work (not thread safe),
        will not be modified in the scope of `self.track`.
        """

    def __len__(self) -> int:
        "The number of items, in the scope of this session."

        return len(self._COST_CUMSUM) - self._before_count

    def __getitem__(self, idx: int | slice[int, int, None], /) -> Cost:
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)

        if idx.step is not None:
            raise ValueError(
                "Only support contiguous cost views right now. "
                "Do not specify the step for slices."
            )

        return self.__getitem_slice(idx.start, idx.stop)

    def __getitem_slice(self, start: int, end: int):
        if not 0 <= start <= len(self):
            raise IndexError

        if not 0 <= end <= len(self):
            raise IndexError

        end_cost = self._COST_CUMSUM[end + self._before_count - 1]
        start_cost = self._COST_CUMSUM[start + self._before_count - 1]
        return end_cost - start_cost

    def sum(self) -> Cost:
        "Return the sum of costs in this current session."
        return self[0 : len(self)]

    def record(self, cost: Cost):
        "Record the `cost` into the session."

        cumsum = self._COST_CUMSUM.top() + cost
        self._COST_CUMSUM.append(cumsum)

    def cleanup(self) -> None:
        assert len(self) >= 0
        self._COST_CUMSUM.truncate(self._before_count)


@ctxl.contextmanager
def _set_latest_session(sess: CostSession):
    "Set the latest session and restore later."

    global _latest_session
    before = _latest_session
    _latest_session = sess

    try:
        yield
    finally:
        _latest_session = before


@ctxl.contextmanager
def track_cost() -> cabc.Generator[CostSession]:
    """
    Track the costs in the `CostSession`.
    """

    sess = CostSession()

    with _set_latest_session(sess):
        try:
            yield sess
        finally:
            sess.cleanup()


def current_session() -> CostSession | None:
    "Get the currently active session."

    return _latest_session
