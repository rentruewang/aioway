# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import loguru as L
import torch

from aioway.t import is_real

__all__ = ["VarInfo", "VarList"]


class VarInfo:
    "Variable information in the DAG."

    def __init__(self, producer: int, fake: torch.Tensor):
        self._producer = producer
        self._fake = fake

        self._consumers: set[int] = set()

        if not isinstance(fake, torch.Tensor) or is_real(fake):
            raise ValueError(
                f"The fake tensor produced at idx={self.producer} is real."
            )

    def __hash__(self) -> int:
        return id(self.fake)

    def add_consumers(self, *consumers: int) -> None:
        "Add consumers for the info."

        if len(set(consumers)) != len(consumers):
            raise ValueError("Duplicate values in consumers.")

        for consumer in consumers:
            self._add_consumer(consumer)

    def _add_consumer(self, consumer: int) -> None:
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
    def is_input(self) -> bool:
        "Check if the variable is an input."
        return self._producer < 0

    @property
    def is_output(self) -> bool:
        "Check if the variable is an output."
        return not self.consumers

    @property
    def alive_until(self) -> int:
        "Get the last step where the variable is alive."

        return max(self.consumers)

    @classmethod
    def input_var(cls, fake: torch.Tensor) -> typing.Self:
        return cls(producer=-1, fake=fake)


@dcls.dataclass(frozen=True)
class _ByStep:
    producers: list[VarInfo] = dcls.field(default_factory=list)
    consumers: list[VarInfo] = dcls.field(default_factory=list)


class VarList:
    """
    The set of variable infos. Almost a mapping but does not support iteration.
    """

    def __init__(self, vars: cabc.Iterable[VarInfo]) -> None:
        vars = tuple(vars)

        L.logger.trace("Attempting to create a stash of {} local vars", len(vars))

        self._vars = {id(var.fake): var for var in vars}

        if len(vars) != len(self._vars):
            raise ValueError("The fake tensors are not unique.")

        self._by_step = self._compute_consumers_by_step()
        "The graph part of the variables."

    def __len__(self) -> int:
        return len(self._vars)

    def __contains__(self, fake: torch.Tensor | int) -> bool:
        return _get_id(fake) in self._vars

    def __iter__(self) -> cabc.Iterator[int]:
        return iter(self._vars)

    def __getitem__(self, fake: torch.Tensor | int) -> VarInfo:
        return self._vars[_get_id(fake)]

    def values(self):
        return self._vars.values()

    def consumers(self, step: int) -> cabc.Sequence[VarInfo]:
        return self._by_step[step].consumers

    def producers(self, step: int) -> cabc.Sequence[VarInfo]:
        return self._by_step[step].producers

    def _compute_consumers_by_step(self) -> dict[int, _ByStep]:
        result: dict[int, _ByStep] = collections.defaultdict(_ByStep)

        for var in self.values():
            result[var.producer].producers.append(var)

            for consumer in var.consumers:
                result[consumer].consumers.append(var)

        return result


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    typing.assert_never(tensor)
