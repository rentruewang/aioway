# Copyright (c) AIoWay Authors - All Rights Reserved

"Domains of tensorclasses: pairs."

import dataclasses as dcls

import tensordict as td
import torch

from aioway.t import parse_attr, tcol_to_tdict

from .domains import AnyDomain, Domain, InvalidValueForDomain
from .tensors import TensorDomain

__all__ = [
    "ScalarDomain",
    "IntTensorDomain",
    "FloatTensorDomain",
    "ScoreTensorDomain",
    "Pair",
    "PairDomain",
    "TDictDomain",
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


class IntTensorDomain(TensorDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if parse_attr(obj).dtype.is_floating_point:
            raise InvalidValueForDomain("Must be an int tensor.")


class FloatTensorDomain(TensorDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

    def check_type(self, obj: torch.Tensor, /) -> None:
        if not parse_attr(obj).dtype.is_floating_point:
            raise InvalidValueForDomain("Must be a float tensor.")


class ScoreTensorDomain(TensorDomain):
    @property
    def __extends__(self) -> list[Domain]:
        return [TensorDomain()]

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
    "A `Pair` whose `input` and `target` are in these domains."

    input: Domain = AnyDomain()
    "The domain of `input`."

    target: Domain = AnyDomain()
    "The domain of `target`."

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def check_type(self, obj: Pair, /) -> None:
        super().check_type(obj)
        if not isinstance(obj, Pair):
            raise InvalidValueForDomain(f"expected a Pair, got {type(obj).__name__}")

        self.input.check_type(obj.input)
        self.target.check_type(obj.target)

    def check_data(self, obj: Pair, /) -> None:
        super().check_data(obj)
        self.input.check_data(obj.input)
        self.target.check_data(obj.target)


class TDictDomain(Domain[td.TensorDictBase]):
    "A `TensorDict` whose entries are in these domains. Other keys are allowed."

    domains: dict[str, Domain] = dcls.field(default_factory=dict)
    "The domain of each key."

    def __hash__(self) -> int:
        return hash(frozenset(self.domains.items()))

    @property
    def __extends__(self) -> list[Domain]:
        return [AnyDomain()]

    def __lt__(self, domain: Domain) -> bool:
        # Same keys, each entry a sub domain.
        if isinstance(domain, TDictDomain) and self != domain:
            return (
                True
                and self.domains.keys() == domain.domains.keys()
                and all(val <= domain.domains[key] for key, val in self.domains.items())
            )

        return super().__lt__(domain)

    def check_type(self, obj: td.TensorDictBase, /) -> None:
        super().check_type(obj)

        if not td.is_tensor_collection(obj):
            raise InvalidValueForDomain(f"expected a TensorDict, got {type(obj)=}")

        # Just so it handles tensorclass as well.
        obj = tcol_to_tdict(obj)

        for key, domain in self.domains.items():
            if key not in obj.keys():
                raise InvalidValueForDomain(f"missing key {key!r}")

            domain.check_type(obj[key])

    def check_data(self, obj: td.TensorDictBase, /) -> None:
        super().check_data(obj)
        for key, domain in self.domains.items():
            domain.check_data(obj[key])
