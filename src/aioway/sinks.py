# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import dataclasses as dcls
import typing

from aioway.hop import Iter

__all__ = ["Sink", "sink_dcls"]


@typing.dataclass_transform(frozen_default=True)
def sink_dcls(cls):
    return dcls.dataclass(frozen=True)(cls)


@sink_dcls
class Sink[T = typing.Any](abc.ABC):
    """
    Consumes a `Iter` and writes to some external location.
    """

    TYPE: typing.ClassVar[type[T]]
    """
    The type to check
    """

    def __call__(self, hop: Iter[T]) -> None:
        for batch in hop:
            if not isinstance(batch, self.TYPE):
                raise TypeError(f"The batch has {type(batch)=}, expected {self.TYPE}.")

            self.write(batch)

    @abc.abstractmethod
    def write(self, batch: T, /) -> None:
        raise NotImplementedError
