# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch

from aioway.nn import TSpec

__all__ = ["Task", "BatchIter"]

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
    def fake(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> BatchIter[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, batch: T, /) -> None:
        raise NotImplementedError
