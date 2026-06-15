# Copyright (c) AIoWay Authors - All Rights Reserved

"The interface for the `io` package."

import abc
import typing

from aioway.hop import Hop

from .dsets import Dset, dset_dcls

__all__ = ["Sink"]


@dset_dcls
class Sink[T = typing.Any](Dset, abc.ABC):
    """
    Consumes a `Hop` and writes to some external location.
    """

    TYPE: typing.ClassVar[type[T]]
    """
    The type to check
    """

    def __call__(self, hop: Hop[T]) -> None:
        for batch in hop:
            if not isinstance(batch, self.TYPE):
                raise TypeError(f"The batch has {type(batch)=}, expected {self.TYPE}.")

            self.write(batch)

    @abc.abstractmethod
    def write(self, batch: T) -> None:
        raise NotImplementedError

    @typing.override
    def _register(self) -> None:
        from .sess import SinkSession

        if sess := SinkSession.current():
            sess.push(self)
