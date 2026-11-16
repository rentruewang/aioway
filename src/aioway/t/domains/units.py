# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of tensorclasses: pairs."

import tensordict as td
import torch

from aioway.t import Attr, parse_attr

from .domains import AnyDomain, Domain, InvalidValueForDomain

__all__ = [
    "TensorDomain",
    "IntTensorDomain",
    "FloatTensorDomain",
    "ScoreTensorDomain",
    "Pair",
    "PairDomain",
]


class TensorDomain(Domain[torch.Tensor]):
    "Any tensor."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if not isinstance(obj, torch.Tensor):
            raise InvalidValueForDomain(f"expected a tensor, got {type(obj).__name__}")


class IntTensorDomain(TensorDomain):
    def check_type(self, obj: torch.Tensor, /) -> None:
        if parse_attr(obj).dtype.is_floating_point:
            raise InvalidValueForDomain("Must be an int tensor.")


class FloatTensorDomain(TensorDomain):
    def check_type(self, obj: torch.Tensor, /) -> None:
        if not parse_attr(obj).dtype.is_floating_point:
            raise InvalidValueForDomain("Must be a float tensor.")


class ScoreTensorDomain(TensorDomain):
    def check_type(self, obj: torch.Tensor, /) -> None:
        super().check_type(obj)
        if obj.numel() != 1:
            raise InvalidValueForDomain("Not a score.")


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
