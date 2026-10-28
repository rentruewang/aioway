# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn
from torch.utils import data as du

__all__ = ["Task"]


class Task[T](typing.Protocol):
    """
    The task API where all subtasks must follow this spec.

    The API is designed to be as flexible as possible.
    """

    def __call__(self, dset: du.Dataset, module: nn.Module) -> T: ...


class Step[B, T](typing.Protocol):
    """
    The training step function.

    This is the batch version of `Task`.
    """

    def __call__(self, batch: B, module: nn.Module) -> T: ...
