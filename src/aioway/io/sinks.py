# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import typing

from aioway.relalg import Exec

__all__ = ["Sink"]


class Sink[T = typing.Any](abc.ABC):
    """
    Consumes a `Exec` and writes to some external location.
    """

    TYPE: typing.ClassVar[type[T]]
    """
    The type to check
    """

    def __call__(self, hop: Exec[T]) -> None:
        for batch in hop:
            if not isinstance(batch, self.TYPE):
                raise TypeError(f"The batch has {type(batch)=}, expected {self.TYPE}.")

            self.write(batch)

    @abc.abstractmethod
    def write(self, batch: T, /) -> None:
        raise NotImplementedError
