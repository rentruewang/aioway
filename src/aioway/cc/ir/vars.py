# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
import dataclasses as dcls
import typing
from collections import abc as cabc

import loguru as L
import torch
from torch.utils import _pytree as pytree

from aioway.torch import is_fake, is_fake_tensor, is_real, is_real_tensor, parse_attr

__all__ = ["VarInfo", "VarScope"]


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

        self._vars = {_get_fake_id(var.fake): var for var in vars}

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
        return self._vars[_get_fake_id(fake)]

    def values(self):
        return self._vars.values()

    def consumers(self, step: int) -> list[VarInfo]:
        return self._by_step[step].consumers

    def producers(self, step: int) -> list[VarInfo]:
        return self._by_step[step].producers

    def _compute_consumers_by_step(self) -> dict[int, _ByStep]:
        result: dict[int, _ByStep] = collections.defaultdict(_ByStep)

        for var in self.values():
            result[var.producer].producers.append(var)

            for consumer in var.consumers:
                result[consumer].consumers.append(var)

        return result


class VarScope(cabc.Mapping[torch.Tensor, torch.Tensor | None]):
    """
    Stores all the local vars that the DAG executes, by their fake tensors.
    It stores the real tensors associated with the fakes in a `VarInfo`,
    and manage the lifetime of fake tensors by `update` / `expire`.

    It acts as a mapping of all fake tensors in the scope,
    but `__getitem__` would be `None` if the tensor is not alive.

    Tracks the currently in scope tensors.
    """

    def __init__(self, vars: cabc.Iterable[VarInfo]) -> None:
        self._vars = VarList(vars)

        L.logger.opt(lazy=True).trace(
            "Attempting to create a stash of {} local vars", self._vars.__len__
        )

        self._alive: dict[int, torch.Tensor] = {}
        "The tensor that is alive, indexed by their fake tensors' ids."

    def __len__(self) -> int:
        """
        Count the total variables tracked.
        """

        return len(self._vars)

    def __contains__(self, tensor) -> bool:
        if not is_fake_tensor(tensor):
            return False

        return id(tensor) in self._vars

    def __iter__(self) -> cabc.Generator[torch.Tensor]:
        for val in self._vars.values():
            yield val.fake

    def __getitem__(self, fake: torch.Tensor) -> torch.Tensor | None:
        if not is_fake_tensor(fake):
            raise KeyError("Input is not fake.")

        return self._alive.get(id(fake))

    def map[T: typing.Any = typing.Any](self, fake: T) -> T:
        """
        Map the values in `fake` to real values.

        This tolerates real tensors in the input.
        """

        return pytree.tree_map_only(torch.Tensor, func=self._map_maybe_fake, tree=fake)

    def update[T: typing.Any = typing.Any](self, fake: T, real: T) -> None:
        """
        Update the fake values to their corresponding real values.

        Both are guaranteed to have the same structure.
        """

        fake_list, fake_struct = pytree.tree_flatten(fake, is_leaf=is_fake_tensor)
        real_list, real_struct = pytree.tree_flatten(real, is_leaf=is_real_tensor)

        if fake_struct != real_struct:
            raise ValueError(
                "The real and fake provided does not have the same structure."
            )

        # Since having same structure.
        assert len(fake_list) == len(real_list)

        # Update every reference. If real tensor exists in `fake_list`, skip.
        for fake_tensor, real_tensor in zip(fake_list, real_list):
            self._update_maybe_fake(fake_tensor, real_tensor)

    def expire(self, step: int) -> None:
        """
        Expire those variables whose step = `step` (`step` must be positive).
        """

        for var in self._vars.consumers(step):
            if var.alive_until == step:
                self.drop(var.fake)

    def clear(self) -> None:
        """
        Clear all the temporary storage for the next run.
        """

        for var in self._vars.values():
            if self.is_alive(var.fake):
                self.drop(var.fake)

    def attach(self, fake: torch.Tensor, real: torch.Tensor) -> None:
        """
        Associate the real tensor with the fake tensor.
        """

        if not is_fake_tensor(fake):
            raise KeyError(f"Key: {type(fake)=} is not fake tensor.")

        if not is_real_tensor(real):
            raise ValueError(f"Value: {type(real)=} is not real tensor.")

        if (fake_attr := parse_attr(fake)) != (real_attr := parse_attr(real)):
            raise ValueError(
                f"Fake {fake_attr} and real {real_attr} have different `Attr` (incompatible)."
            )

        # Store the real tensor in scope.
        self._alive[_get_fake_id(fake)] = real

    def drop(self, fake: torch.Tensor) -> None:
        """
        Drop a tensor that is currently in scope.
        """

        del self._alive[_get_fake_id(fake)]

    def value(self, fake: torch.Tensor) -> torch.Tensor:
        """
        Get the real value of the fake tensor.
        """

        return self._alive[_get_fake_id(fake)]

    def is_alive(self, obj: torch.Tensor | VarInfo, /) -> bool:
        if isinstance(obj, VarInfo):
            obj = obj.fake

        return _get_fake_id(obj) in self._alive

    def tracked(self) -> cabc.Generator[torch.Tensor]:
        "Get all the fake tensors tracked."

        for info in self._vars.values():
            yield info.fake

    def info(self, tensor: int | torch.Tensor) -> VarInfo:
        "Check if the tensor is tracked."

        if isinstance(tensor, torch.Tensor):
            tensor = id(tensor)

        return self._vars[tensor]

    def _map_maybe_fake(self, item: torch.Tensor) -> torch.Tensor:
        if is_real_tensor(item):
            return item

        # Allow some fake tensors not to be mapped.
        fake_id = _get_fake_id(item)
        return self._alive.get(fake_id, item)

    def _update_maybe_fake(self, fake_tensor: torch.Tensor, real_tensor: torch.Tensor):
        if is_fake_tensor(fake_tensor):
            self.attach(fake_tensor, real_tensor)
            return

        if fake_tensor is not real_tensor:
            raise ValueError("Real tensor in `fake` paired with a different value.")

    def _compute_consumers_by_step(self) -> dict[int, list[VarInfo]]:
        result: dict[int, list[VarInfo]] = collections.defaultdict(list)

        for var in self._vars.values():
            for consumer in var.consumers:
                result[consumer].append(var)

        return result


def _get_id(tensor: torch.Tensor | int, /) -> int:
    match tensor:
        case torch.Tensor():
            return id(tensor)
        case int():
            return tensor

    typing.assert_never(tensor)


def _get_fake_id(fake: torch.Tensor | int, /) -> int:
    if isinstance(fake, torch.Tensor):
        assert is_fake(fake)

    return _get_id(fake)
