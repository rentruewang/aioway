# Copyright (c) AIoWay Authors - All Rights Reserved

"The main `Env` API."

import abc

from aioway.errors import re_raise_func
from aioway.spaces import DataSpace

__all__ = ["Env", "EnvObserv", "RlEnvObserv"]


class Env[O, A, R](abc.ABC):
    """
    The environment protocol defines how I/O could be consumed.

    This is already a `VecEnv`.

    This is similar to `gym.Env` but with a cleaner generator interface.

    This does not support reset, as we favor a new generator everytime,
    and persistent info should be handled by the instance that generates an `Env`.
    """

    @re_raise_func(AssertionError, ValueError)
    def reset(self) -> O:
        state = self._reset()
        assert state in self.observ_space
        return state

    @re_raise_func(AssertionError, ValueError)
    def step(self, action: A, /) -> O:
        assert action in self.action_space

        state = self._step(action)
        assert state in self.observ_space
        return state

    @abc.abstractmethod
    def _reset(self) -> O:
        raise NotImplementedError

    @abc.abstractmethod
    def _step(self, action: A, /) -> O:
        "The step implementaiton. Raise `StopIteration` when done."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observ_space(self) -> DataSpace:
        "The observation space."
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self) -> DataSpace:
        "The action space."
        raise NotImplementedError
