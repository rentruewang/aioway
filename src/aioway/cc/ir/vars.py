# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
import typing
from collections import abc as cabc

import loguru as L
import torch
from torch.utils import _pytree as pytree

from aioway.torch import is_fake, is_fake_tensor, is_real, is_real_tensor

__all__ = ["VarInfo", "LocalVars"]


class VarInfo:
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
    def tensor(self) -> torch.Tensor:
        "The tensor. `self.is_alive` must be true, or `RuntimeError` is raised."

        if self._tensor is None:
            raise AttributeError("No tensor set.")

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
            raise AttributeError("Tensor is already dead. Cannot remove again.")

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
    def alive_until(self) -> int:
        "Get the last step where the variable is alive."

        return max(self.consumers)


class LocalVars:
    "The locals stash, storing all the local variables."

    def __init__(self, variables: cabc.Sequence[VarInfo]) -> None:
        self._variables = variables

        L.logger.opt(lazy=True).trace(
            "Attempting to create a stash of {} local vars", self._variables.__len__
        )

        self._fake_index = self._compute_fake_index()
        "Mapping from id of fake tensor to variable info."

        self._consumers = self._compute_consumers()
        "Mapping from DAG index of consuming point to corresponding variable info."

    def __len__(self) -> int:
        return len(self._variables)

    def __getitem__(self, fake: torch.Tensor) -> torch.Tensor:
        if not is_fake_tensor(fake):
            raise KeyError("Input is not fake.")

        info = self._fake_index[id(fake)]

        try:
            return info.tensor
        except AttributeError:
            raise KeyError("The tensor is not set for this key.")

    def __setitem__(self, fake: torch.Tensor, real: torch.Tensor) -> None:
        if not is_fake_tensor(fake):
            raise KeyError(f"Key: {type(fake)=} is not fake tensor.")

        if not is_real_tensor(real):
            raise ValueError(f"Value: {type(real)=} is not real tensor.")

        var_info = self._fake_index[id(fake)]
        assert not var_info.is_alive

        # Store the real tensor onto the info.
        var_info.tensor = real

    def map(self, fake) -> typing.Any:
        """
        Map the values in `fake` to real values.

        This tolerates real tensors in the input.
        """

        return pytree.tree_map_only(torch.Tensor, func=self._map_maybe_fake, tree=fake)

    def update(self, fake, real) -> None:
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

        for var in self._consumers[step]:
            if var.alive_until == step:
                del var.tensor

    def _compute_fake_index(self) -> dict[int, VarInfo]:
        # Since fake tensors have 1 single producer, the producer is unique.
        result = {id(info.fake): info for info in self._variables}

        if len(result) != len(self._variables):
            raise ValueError("The producer for the variable list is not unique.")

        return result

    def _compute_consumers(self) -> dict[int, list[VarInfo]]:
        result: dict[int, list[VarInfo]] = collections.defaultdict(list)

        for var in self._variables:
            for consumer in var.consumers:
                result[consumer].append(var)

        return result

    def _map_maybe_fake(self, item: torch.Tensor) -> torch.Tensor:
        if is_real_tensor(item):
            return item

        return self[item]

    def _update_maybe_fake(self, fake_tensor: torch.Tensor, real_tensor: torch.Tensor):
        if is_fake_tensor(fake_tensor):
            self[fake_tensor] = real_tensor
            return

        if fake_tensor is not real_tensor:
            raise ValueError("Real tensor in `fake` paired with a different value.")
