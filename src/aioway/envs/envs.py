# Copyright (c) AIoWay Authors - All Rights Reserved

"The main `Env` API."

import abc

from aioway.envs import EnvState
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

    type _EnvState = EnvState[O]
    "Alias of `EnvState` in the class to save some typing."

    @abc.abstractmethod
    def snapshot(self) -> _EnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def restore(self, state: _EnvState) -> None:
        raise NotImplementedError

    # The state property.
    # Using a "forwarding" method s.t. `snapshot`, `restore` can be polymorphic
    def __snapshot(self) -> _EnvState:
        return self.snapshot()

    def __restore(self, state: _EnvState):
        self.restore(state)

    state = property(__snapshot, __restore)
    "The state of the `Env`"

    def observ(self):
        observ = self.state.observ()
        assert observ in self.observ_space
        return observ

    @re_raise_func(AssertionError, ValueError)
    def step(self, action: A, /) -> R:
        assert action in self.action_space

        reward = self._step(action)
        return reward

    @abc.abstractmethod
    def _step(self, action: A, /) -> R:
        """
        The step implementaiton.

        Args:
            action: The action to accept.

        Returns:
            The reward, if any.

            e.g. This is `None` in supervised learning algorithms).

        Raises:
            StopIteration: When done.
        """
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
