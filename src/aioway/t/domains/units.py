# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of tensorclasses: pairs."

import tensordict as td
import torch

from .domains import AnyDomain, Domain, InvalidDomain

__all__ = ["TensorDomain", "Pair", "PairDomain"]


class TensorDomain(Domain[torch.Tensor]):
    "Any tensor."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if not isinstance(obj, torch.Tensor):
            raise InvalidDomain(f"expected a tensor, got {type(obj).__name__}")


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
            raise InvalidDomain(f"expected a Pair, got {type(obj).__name__}")
