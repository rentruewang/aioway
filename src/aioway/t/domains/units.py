# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of tensorclasses: pairs."

import tensordict as td
import torch

from aioway.t import Attr

from .domains import AnyDomain, Domain, InvalidValueForDomain

__all__ = [
    "ScalarDomain",
    "Pair",
    "PairDomain",
]


class ScalarDomain[T](Domain[T]):
    "A domain that wraps a scalar type."

    base: type[T]
    "The base type."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj) -> None:
        if not isinstance(obj, self.base):
            raise InvalidValueForDomain(f"Expected {self.base}, got {type(obj)=}.")


class Pair(td.TensorClass):
    "An input and its target: what a split gives."

    input: torch.Tensor
    """
    What the model sees.
    """

    target: torch.Tensor
    """
    What the model should predict from `input`.
    """


class PairDomain(Domain[Pair]):
    "Any `Pair`."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: Pair, /) -> None:
        super().check_type(obj)
        if not isinstance(obj, Pair):
            raise InvalidValueForDomain(f"expected a Pair, got {type(obj).__name__}")
