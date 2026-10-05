# Copyright (c) AIoWay Authors - All Rights Reserved

import abc

from torch import nn

from aioway.t import TSpec

__all__ = ["Task"]


class Task[T](abc.ABC):
    """
    The task API where all subtasks must follow this spec.

    The API is designed to be as flexible as possible.
    """

    def __call__(self, data, module: nn.Module) -> T:
        self.input_spec.assert_is_in(data)
        result = module(data)
        self.output_spec.assert_is_in(result)
        return result

    @property
    @abc.abstractmethod
    def input_spec(self) -> TSpec:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_spec(self) -> TSpec:
        raise NotImplementedError
