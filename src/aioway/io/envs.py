# Copyright (c) AIoWay Authors - All Rights Reserved

import copy
import abc
import enum
import typing
from collections import abc as cabc

from aioway.errors import re_raise_func
from aioway.spaces import DataSpace, ModuleSpace

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
    def __space__(self) -> ModuleSpace:
        """
        The space that this `Env` satisfies.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _get_next(self) -> Y:
        raise NotImplementedError

    @property
    def observ_space(self) -> DataSpace:
        return self.__space__().observ_space

    @property
    def action_space(self) -> DataSpace:
        return self.__space__().action_space

    @re_raise_func(AssertionError, ValueError)
    def _check_observation(self, observation: Y, /) -> None:
        assert observation in self.observ_space

    @abc.abstractmethod
    def clone(self) -> typing.Self:
        """
        Clone the current `Env`. By default calls `copy.copy`.
        Subclass should overwrite the behavior.
        """
        return copy.copy(self)

    @property
    def status(self) -> EnvStatus:
        "The status of the environment."

        return self._status

    def _status_start(self):
        if self._status is EnvStatus.PENDING:
            self._status = EnvStatus.RUNNING

    def _status_done(self):
        self._status = EnvStatus.FINISHED


class ActionableEnv[Y, S, R](Env[Y], cabc.Generator[Y, S, R]):
    """
    The `Env` that supports `.send`.

    This is similar to `gym.Env` but with a cleaner generator interface.

    This does not support reset, as we favor a new generator everytime,
    and persistent info should be handled by the instance that generates an `Env`.
    """

    @re_raise_func(AssertionError, ValueError)
    def send(self, action: S, /) -> Y:
        assert action in self.action_space
        observ = self._step(action)
        assert observ in self.observ_space
        return observ

    @abc.abstractmethod
    def _get_first(self) -> Y:
        raise NotImplementedError

    @abc.abstractmethod
    def _step(self, action: S, /) -> Y:
        raise NotImplementedError

    @abc.abstractmethod
    def throw(self, typ, val=None, tb=None) -> Y:
        raise NotImplementedError
