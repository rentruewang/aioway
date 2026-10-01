# Copyright (c) AIoWay Authors - All Rights Reserved

"Executing the DAG with new data."

import typing
from collections import abc as cabc

import torch
from torch.utils import _pytree as pytree

from aioway.t import is_fake, is_fake_tensor, is_real_tensor, parse_attr

from .sets import InstrSet, FCall

__all__ = ["Exec", "LocalScope"]


class LocalScope(cabc.Mapping[torch.Tensor, torch.Tensor | None]):
    """
    Stores all the local vars that the DAG executes, by their fake tensors.
    It stores the real tensors associated with the fakes in a `VarInfo`,
    and manage the lifetime of fake tensors by `update` / `expire`.

    It acts as a mapping of all fake tensors in the scope,
    but `__getitem__` would be `None` if the tensor is not alive.

    Tracks the currently in scope tensors.
    """

    def __init__(self, dag: InstrSet) -> None:
        self._dag = dag
        "The variable list."

        self._alive: dict[int, torch.Tensor] = {}
        "The tensor that is alive, indexed by their fake tensors' ids."

    def __len__(self) -> int:
        """
        Count the total variables tracked.
        """

        return len(self._dag)

    def __contains__(self, tensor) -> bool:
        if not is_fake_tensor(tensor):
            return False

        return tensor in self._dag.tensors

    def __iter__(self) -> cabc.Generator[torch.Tensor]:
        yield from self._dag.tensors

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

        fake_list = pytree.tree_leaves(fake, is_leaf=is_fake_tensor)
        real_list = pytree.tree_leaves(real, is_leaf=is_real_tensor)

        if len(fake_list) != len(real_list):
            raise ValueError(
                "The real and fake provided does not have the same structure."
            )

        if any(parse_attr(f) != parse_attr(t) for f, t in zip(fake_list, real_list)):
            raise ValueError("The parsed lists are not really compatible.")

        # Update every reference. If real tensor exists in `fake_list`, skip.
        for fake_tensor, real_tensor in zip(fake_list, real_list):
            self._update_maybe_fake(fake_tensor, real_tensor)

    def expire(self, step: int) -> None:
        """
        Expire those variables that is not used after step = `step`.
        """

        thunk = self._dag[step]

        for input in thunk.inputs:
            if self._dag.life(input).death == step:
                self.drop(input)

    def clear(self) -> None:
        """
        Clear all the temporary storage for the next run.
        """

        self._alive = {}

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

    def is_alive(self, obj: torch.Tensor, /) -> bool:
        return _get_fake_id(obj) in self._alive

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


class Exec[F: cabc.Callable = typing.Any]:
    """
    This is the DAG executor responsible for executing a traced thunk list on real data.
    """

    def __init__(self, dag: InstrSet) -> None:
        self._dag = dag
        self._scope = LocalScope(self._dag)

    def __len__(self) -> int:
        return len(self._dag)

    def __getitem__(self, idx: int) -> FCall[F]:
        return self._dag[idx]

    def __iter__(self) -> cabc.Generator[FCall[F]]:
        yield from self._dag

    def __call__(self, *inputs: torch.Tensor) -> typing.Any:
        try:
            self._scope.update(self.inputs, inputs)
        except ValueError as err:
            raise TypeError from err

        # Execute the steps one by one in topo sorted order.
        for idx, thunk in enumerate(self._dag):
            args, kwargs = self._scope.map([thunk.args, thunk.kwargs])
            real = thunk.func(*args, **kwargs)
            self._scope.update(thunk.result, real)
            self._scope.expire(idx)

        result = self._scope.map(self._dag[-1].result)
        self._scope.clear()
        return result

    @property
    def inputs(self):
        return self._dag.inputs


def _get_fake_id(fake: torch.Tensor | int, /) -> int:
    if isinstance(fake, torch.Tensor):
        assert is_fake(fake)
        return id(fake)

    if isinstance(fake, int):
        return fake

    raise TypeError(fake)
