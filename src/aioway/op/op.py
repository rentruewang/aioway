# Copyright (c) AIoWay Authors - All Rights Reserved

"The operator base class."

import abc
import torch

__all__ = []


class Op(abc.ABC):
    """
    `Op` stands for [op]erator, or [o]peration [p]review.
    It is essentailly an unevaluated expression that supports inspection.
    """

    @abc.abstractmethod
    def do(self) -> torch.Tensor:
        raise NotImplementedError
