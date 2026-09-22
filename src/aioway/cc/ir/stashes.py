# Copyright (c) AIoWay Authors - All Rights Reserved

from collections import abc as cabc

import torch

from aioway.torch import is_fake, is_real

__all__ = ["DagVarInfo"]


class DagVarInfo:
    "Variable information in the DAG."

    def __init__(self, producer: int, fake: torch.Tensor):
        self._producer = producer

        self._consumers: list[int] = []

        self._fake = fake

        self._tensor: torch.Tensor | None = None

        if not isinstance(fake, torch.Tensor) or is_real(fake):
            raise ValueError(
                f"The fake tensor produced at idx={self.producer} is real."
            )

    def __hash__(self) -> int:
        return id(self.fake)

    @property
    def producer(self) -> int:
        "The producer index."
        return self._producer

    @property
    def consumers(self) -> cabc.Sequence[int]:
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
