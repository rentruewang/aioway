# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import enum
import typing
from collections import abc as cabc

from aioway.errors import re_raise_func
from aioway.spaces import DataSpace

__all__ = ["EnvStatus", "Env", "ResetEnv"]


class EnvStatus(enum.StrEnum):
    """
    The status of the environment.
    """

    PENDING = enum.auto()
    "The environment is not started yet."

    RUNNING = enum.auto()
    "The environment is running."

    FINISHED = enum.auto()
    "The environment is done executing."


class Env[Y](cabc.Iterator[Y], abc.ABC):
    """
    The environment protocol defines how I/O could be consumed.
    """

    def __init__(self) -> None:
        self._status = EnvStatus.PENDING

    def __iter__(self) -> typing.Self:
        return self

    def __next__(self) -> Y:
        self._status_start()

        try:
            return self._get_next()
        except StopIteration:
            # Set to finish after done.
            self._status_done()
            raise

    @abc.abstractmethod
    def _get_next(self) -> Y:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observ_space(self) -> DataSpace:
        "Observation space."

        raise NotImplementedError

    @re_raise_func(AssertionError, ValueError)
    def _check_observation(self, observation: Y, /) -> None:
        assert observation in self.observ_space

    @property
    def status(self) -> EnvStatus:
        "The status of the environment."

        return self._status

    def _status_start(self):
        if self._status is EnvStatus.PENDING:
            self._status = EnvStatus.RUNNING

    def _status_done(self):
        self._status = EnvStatus.FINISHED


class ResetEnv[Y](Env[Y], abc.ABC):
    """
    This is an environment that supports resetting.

    After calling `.reset`, a `next` call is needed to restart the `Env`.
    """

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
