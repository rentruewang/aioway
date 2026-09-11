# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.nn import TSpec

__all__ = ["NnInput", "Task", "BatchIter"]

type NnInput = torch.Tensor | td.TensorClass | td.TensorDict


class BatchIter[T: NnInput = typing.Any](typing.Protocol):
    """
    The constraints that trainer uses to define what trainer handles.
    """

    def __iter__(self) -> cabc.Iterator[T]:
        "Iterates and yield batches."

        ...

    def __tspec__(self) -> TSpec:
        """
        The space constraining the output of `__iter__`.
        """


class Task[T: NnInput](abc.ABC):
    @abc.abstractmethod
    def fake(self) -> T:
        """
        Generate one batch of fake data.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> BatchIter[T]:
        """
        Yield the batches required to run the task.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def step(self, batch: T, /) -> None:
        """
        Perform each step in the task.
        In `fake_mode`, it should only do cheap operations.
        """

        raise NotImplementedError

    def fake_step(self) -> None:
        """
        Step once, with fake data.
        Useful for tracking computation symbolically.
        """

        fake_data = self.fake()
        self.step(fake_data)
