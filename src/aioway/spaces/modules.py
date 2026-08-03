# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing

from torch import nn

from aioway._tensors import torch_fake_mode

from .data import DataSpace
from .spaces import Space, space_dcls

__all__ = ["ModuleSpace", "StatelessSpace"]


@space_dcls
class ModuleSpace[O, A](Space[nn.Module], abc.ABC):
    """
    An spec accepts actions (outputs from the models),
    and observations (inputs to the models).

    It's the environemnt in RL, but adapted to also work with supervised learning etc,
    by using the `Generator` abstraction (using `.send` to send actions).

    The constructor takes the observation space and the action space.
    """

    @typing.override
    def contains(self, module: nn.Module) -> bool:
        with torch_fake_mode():
            return self._check_contains(module)

    def _check_contains(self, module: nn.Module) -> bool:
        input = self.observ_space.sample()
        output = module(input)
        return output in self.action_space

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


@space_dcls
class StatelessSpace(ModuleSpace):
    observ: DataSpace
    "The input of the `nn.Module`."

    action: DataSpace
    "The output of the `nn.Module`."

    @property
    @typing.override
    def observ_space(self) -> DataSpace:
        return self.observ

    @property
    @typing.override
    def action_space(self) -> DataSpace:
        return self.action
