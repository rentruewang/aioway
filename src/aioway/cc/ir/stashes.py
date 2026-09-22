# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
from collections import abc as cabc

import torch

from aioway._utils import AnyDict
from aioway.torch import is_fake, is_real

__all__ = ["DagVarInfo"]


class DagVarInfo:
    "Variable information in the DAG."

    def __init__(self, producer: int, fake: torch.Tensor):
        self._producer = producer

        self._consumers: set[int] = set()

        self._fake = fake

        self._tensor: torch.Tensor | None = None

        if not isinstance(fake, torch.Tensor) or is_real(fake):
            raise ValueError(
                f"The fake tensor produced at idx={self.producer} is real."
            )

    def add_consumer(self, consumer: int) -> None:
        if consumer in self.consumers:
            raise IndexError(f"Attempting to add {consumer=} a second time.")

        self._consumers.add(consumer)

    @property
    def producer(self) -> int:
        "The producer index."
        return self._producer

    @property
    def consumers(self) -> cabc.Set[int]:
        "The list of consumers."
        return self._consumers

    @property
    def fake(self) -> torch.Tensor:
        "Return the fake tensor."
        return self._fake

    @property
    def tensor(self) -> torch.Tensor:
        "The tensor. `self.is_alive` must be true, or `RuntimeError` is raised."

        if self._tensor is None:
            raise RuntimeError("No tensor set.")

        return self._tensor

    @tensor.setter
    def tensor(self, tensor: torch.Tensor) -> None:
        "Setting the tensor to a real tensor."

        if not isinstance(tensor, torch.Tensor):
            raise ValueError("The input tensor is not a tensor.")

        if is_fake(tensor):
            raise ValueError("The input tensor is a fake tensor.")

        self._tensor = tensor

    @tensor.deleter
    def tensor(self) -> None:
        "Free the current tensor."

        if not self.is_alive:
            raise RuntimeError("Tensor is already dead. Cannot remove again.")

        self._tensor = None

    @property
    def is_alive(self) -> bool:
        "Check if this variable has a live tensor associated with it."
        return self._tensor is not None

    @property
    def is_input(self) -> bool:
        "Check if the variable is an input."
        return self._producer < 0

    @property
    def largest_consumer(self) -> int:
        "Get the largest consumer."
        return max(self.consumers)


class DagLocals:
    def __init__(self, variables: cabc.Sequence[DagVarInfo]) -> None:
        self._variables = variables

        self._fake_index = self._compute_fake_index()
        "Mapping from id of fake tensor to variable info."

        self._consumers = self._compute_consumers()
        "Mapping from DAG index of consuming point to corresponding variable info."

    def __len__(self) -> int:
        return len(self._variables)

    def update(self, fake, real) -> None:
        pass

    def fill(self, step: int, *args, **kwargs):
        pass

    def _fill_container(self, step: int, *args, **kwargs):
        pass

    def _post_fill(self, step: int):
        pass

    def _compute_fake_index(self) -> dict[int, DagVarInfo]:
        result = {id(info.fake): info for info in self._variables}
        assert len(result) == len(self._variables)
        return result

    def _compute_consumers(self) -> dict[int, list[DagVarInfo]]:
        result: dict[int, list[DagVarInfo]] = collections.defaultdict(list)

        for var in self._variables:
            for consumer in var.consumers:
                result[consumer].append(var)

        return result
