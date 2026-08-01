# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway.errors import re_raise_func
from aioway.spaces import DataSpace

__all__ = ["Env"]


@dcls.dataclass(frozen=True)
class Env[Y](cabc.Iterator[Y]):
    """
    The environment protocol defines how I/O could be consumed.
    """

    def __iter__(self) -> typing.Self:
        return self

    @abc.abstractmethod
    def __next__(self) -> Y:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observ_space(self) -> DataSpace:
        "Observation space."

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self) -> DataSpace:
        "Action space."

        raise NotImplementedError

    @re_raise_func(AssertionError, ValueError)
    def _check_observation(self, observation: Y, /) -> None:
        assert observation in self.observ_space
