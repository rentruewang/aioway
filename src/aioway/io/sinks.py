# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import typing

__all__ = ["Sink"]


class Sink[T = typing.Any](abc.ABC):
    """
    Consumes a `Exec` and writes to some external location.
    """

    TYPE: typing.ClassVar[type[T]]
    """
    The type to check
    """

    @abc.abstractmethod
    def write(self, batch: T, /) -> None:
        raise NotImplementedError
