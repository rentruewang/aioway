# Copyright (c) AIoWay Authors - All Rights Reserved

"The operator base class."

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["Hop"]


@typing.dataclass_transform(frozen_default=True, kw_only_default=True)
def hop_dcls(cls):
    return dcls.dataclass(frozen=True, kw_only=True)(cls)


@hop_dcls
class Hop[T = object](abc.ABC):
    """
    `Hop` stands for [h]igh level [op]erator, or [h]igh level [o]peration [p]review.
    It is essentailly an unevaluated expression that supports inspection.
    """

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def deps(self) -> cabc.Iterator[Hop]:
        """
        The dependent `Hop`s.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def do(self) -> T:
        """
        Evaluates the current operator and outputs the results.
        The object must be decomposed into pure tensors (no extra items e.g. primitives).
        """

        raise NotImplementedError
