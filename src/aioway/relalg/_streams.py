# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Stream` interfaces live here."

import dataclasses as dcls
import typing

__all__ = ["StreamState"]


@typing.dataclass_transform(frozen_default=True)
def stream_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@dcls.dataclass
class StreamState:
    """
    The mutable stream state.

    This is created because `Stream` subclasses from a frozen `dataclass`,
    so the stream state is created to manage mutable parts of the `Stream`.

    Subclasses of `Stream` should also subclass from `StreamState`.
    """

    idx: int = 0
    "How many steps have been called."

    def step(self):
        self.idx += 1

    @property
    def started(self) -> bool:
        """
        Shortcut function to check if `self.idx == 0`.
        """

        return self.idx != 0
