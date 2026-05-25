# Copyright (c) AIoWay Authors - All Rights Reserved

"A `Space` is a constraint on `Schema`, used by an `Algo` to constrain `Hop`."

import abc
import dataclasses as dcls
import typing

import torch

from aioway._torch.fake import is_fake_tensor
from aioway.schemas import Schema

__all__ = ["Space", "AnySpace", "SchemaSpace"]


@typing.dataclass_transform(frozen_default=True)
def space_dcls(cls):
    return dcls.dataclass(frozen=True, slots=True)(cls)


@space_dcls
class Space(abc.ABC):
    "Mostly a wrapper around a `gymnasium.Space` or items looking like those."

    def __contains__(self, x: Schema | torch.Tensor, /) -> bool:
        if isinstance(x, Schema):
            x = x.to_fake_tensor()

        if isinstance(x, torch.Tensor):
            return self._contains_tensor(x)

        raise TypeError(f"Unsupported {type(x)=}.")

    @abc.abstractmethod
    def _contains_tensor(self, x: torch.Tensor) -> bool:
        raise NotImplementedError


@space_dcls
class AnySpace(Space):
    @typing.override
    def _contains_tensor(self, x: torch.Tensor) -> bool:
        return True


@space_dcls
class SchemaSpace(Space):
    """
    The space inspired by `gym.Box`. Only supports the ndarray version,
    """

    schema: Schema
    low: float = -float("inf")
    high: float = float("inf")

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"{self.low=} > {self.high=}.")

    @typing.override
    def _contains_tensor(self, x: torch.Tensor) -> bool:
        if Schema.from_tensor(x) != self.schema:
            return False

        if is_fake_tensor(x):
            return True

        return bool(torch.all(x >= self.low)) and bool(torch.all(x <= self.high))
