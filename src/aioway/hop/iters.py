# Copyright (c) AIoWay Authors - All Rights Reserved

"The iterator for `Hop`."

import abc
import typing
from collections import abc as cabc

__all__ = ["HopIter", "HopGenIter"]


class HopIter[T = typing.Any](cabc.Iterator, abc.ABC):
    def __init__(self) -> None:
        self.__idx: int = 0

    def __iter__(self):
        return self

    @typing.final
    def __next__(self) -> T:
        # If `StopIteration` is raised here, it's done.
        result = self.read()
        self.__idx += 1
        return result

    @abc.abstractmethod
    def read(self) -> T:
        raise NotImplementedError

    @property
    def idx(self) -> int:
        "Get the current iteration count."

        return self.__idx

    @property
    def started(self) -> bool:
        "Shortcut function to check if `self.idx == 0`."

        return self.idx != 0


class HopGenIter[T = typing.Any](HopIter[T]):
    def __init__(self, generator: cabc.Iterator[T]):
        super().__init__()

        self._generator = generator
        """
        The generator from which to yield.
        """

    @typing.override
    def read(self) -> T:
        return next(self._generator)

    @property
    def generator(self) -> cabc.Iterator[T]:
        return self.generator
