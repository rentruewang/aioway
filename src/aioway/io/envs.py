# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn

from aioway.emits import emit
from aioway.errors import re_raise_func
from aioway.relalg import LoaderOpt
from aioway.spaces import Space

from .dsets import Dset
from .sinks import Sink

__all__ = ["Env", "EnvGen"]


class Env[O = typing.Any, A = typing.Any](abc.ABC):
    """
    An environment accepts actions (outputs from the models),
    and outputs observations (inputs to the models).

    It's the environemnt in RL, but adapted to also work with supervised learning etc,
    by using the `Generator` abstraction (using `.send` to send actions).

    The constructor takes the observation space and the action space.
    """

    def __call__(self) -> cabc.Generator[O, A, None]:
        yield from self.generator()

    def generator(self) -> EnvGen[O, A, None]:
        """
        Wrap the generator, check all its observations and actions.
        """

        return EnvGen(
            observ_space=self.observ_space,
            action_space=self.action_space,
            generator=self.interact(),
        )

    @abc.abstractmethod
    def interact(self) -> cabc.Generator[O, A, None]:
        """
        Yields a generator that can accept actions from the agents,
        or ignores them (`action = yield observation`).
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observ_space(self) -> Space:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self) -> Space:
        raise NotImplementedError

    def emit(self) -> cabc.Generator[nn.Module]:
        yield from emit(self.observ_space, self.action_space)


class IoEnv(Env):
    "An `Env` that supports inputs and outputs."

    def __init__(self, dset: Dset, sink: Sink, opt: LoaderOpt) -> None:
        super().__init__()
        self._dset = dset
        self._sink = sink
        self._opt = opt

    @property
    @typing.override
    def observ_space(self):
        return self._dset.space

    @property
    @typing.override
    def action_space(self):
        return self._sink.space

    @typing.override
    def interact(self):
        for observ in self._dset(self._opt):
            action = yield observ
            self._sink.write(action)


@dcls.dataclass(frozen=True)
class EnvGen[Y, S, R](cabc.Generator[Y, S, R]):
    """
    The environment generator, performing checks while implementing the generator protocol.
    """

    observ_space: Space
    "Observation space."

    action_space: Space
    "Action space."

    generator: cabc.Generator[Y, S, R] = dcls.field(repr=False)
    """
    The generator that `EnvGen` wraps.
    Its inputs (actions) and outputs (observations) would be checked.
    """

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

    @re_raise_func(AssertionError, ValueError)
    def _check_observation(self, observation: Y, /) -> None:
        assert observation in self.observ_space

    @re_raise_func(AssertionError, ValueError)
    def _check_action(self, action: S, /) -> None:
        assert action in self.action_space
