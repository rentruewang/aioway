# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from .spaces import Space

__all__ = ["Env", "EnvGen"]


class Env[O = typing.Any, A = typing.Any](abc.ABC):
    """
    An environment accepts actions (outputs from the models),
    and outputs observations (inputs to the models).

    It's the environemnt in RL, but adapted to also work with supervised learning etc,
    by using the `Generator` abstraction (using `.send` to send actions).

    The constructor takes the observation space and the action space.
    """

    def __init__(self, observation_space: Space, action_space: Space) -> None:
        self._observation_space = observation_space
        self._action_space = action_space

    def __call__(self) -> cabc.Generator[O, A, None]:
        yield from self.generator()

    def generator(self) -> EnvGen[O, A, None]:
        """
        Wrap the generator, check all its observations and actions.
        """

        return EnvGen(
            observation_space=self._observation_space,
            action_space=self._action_space,
            generator=self.interact(),
        )

    @abc.abstractmethod
    def interact(self) -> cabc.Generator[O, A, None]:
        """
        Yields a generator that can accept actions from the agents,
        or ignores them (`action = yield observation`).
        """

        raise NotImplementedError


@dcls.dataclass(frozen=True)
class EnvGen[Y, S, R](cabc.Generator[Y, S, R]):
    """
    The environment generator, performing checks while implementing the generator protocol.
    """

    observation_space: Space
    action_space: Space
    generator: cabc.Generator[Y, S, R] = dcls.field(repr=False)

    def __iter__(self) -> typing.Self:
        return self

    def __next__(self) -> Y:
        observation = next(self.generator)
        self._check_observation(observation)
        return observation

    def send(self, action: S, /) -> Y:
        self._check_action(action)
        observation = self.generator.send(action)
        self._check_observation(observation)
        return observation

    def throw(self, typ, val=None, tb=None) -> Y:
        return self.generator.throw(typ, val, tb)

    def _check_observation(self, observation: Y, /) -> None:
        if observation not in self.observation_space:
            raise ValueError

    def _check_action(self, action: S, /) -> None:
        if action not in self.action_space:
            raise ValueError
