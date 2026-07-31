# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.spaces import Space
import abc
import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from torch import nn


class Spec[O = typing.Any, A = typing.Any](abc.ABC):
    """
    An spec accepts actions (outputs from the models),
    and observations (inputs to the models).

    It's the environemnt in RL, but adapted to also work with supervised learning etc,
    by using the `Generator` abstraction (using `.send` to send actions).

    The constructor takes the observation space and the action space.
    """

    def __contains__(self, module: nn.Module) -> bool:
        input = self.observ_space.sample()
        output = module(input)
        return output in self.action_space

    @property
    @abc.abstractmethod
    def observ_space(self) -> Space:
        "The observation space."

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_space(self) -> Space:
        "The action space."

        raise NotImplementedError
