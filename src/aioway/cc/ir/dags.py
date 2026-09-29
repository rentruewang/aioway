# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import collections
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway.t import (
    TList,
    find_nested_tensors,
    is_real,
    render_tensor_func_short,
    replace_tensors_with_attr,
)

from .vars import VarInfo, VarList

__all__ = ["ThunkNode", "TensorRef", "Dag0"]


@dcls.dataclass(frozen=True)
class ThunkNode[F: cabc.Callable]:
    """
    Stores the thunk's arguments, function, and output.

    This is the node type for the dag.
    """

    _: dcls.KW_ONLY

    func: F
    "The callable that the thunk calls."

    args: tuple[typing.Any, ...]
    "The arguments fed to the function."

    kwargs: dict[str, typing.Any]
    "The keyword arguments fed to the function."

    result: typing.Any = dcls.MISSING
    "The result. If `dcls.MISSING`, the thunk is not called yet."

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError(f"{self.func} is not callable.")

    @typing.override
    def __repr__(self) -> str:
        result = str(replace_tensors_with_attr(self.result))
        thunk = render_tensor_func_short(str(self.func), self.args, self.kwargs)
        return thunk + " -> " + result

    @functools.cached_property
    def upstreams(self) -> TList:
        "Get the (unique) dependencies of the current thunk."
        return TList(self._upstream())

    @functools.cached_property
    def downstreams(self) -> TList:
        "Get the output list of (unique) tensors of the current thunk."
        return TList(self._downstream())

    def _upstream(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.args)
        yield from find_nested_tensors(self.kwargs)

    def _downstream(self) -> cabc.Generator[torch.Tensor]:
        yield from find_nested_tensors(self.result)

    @property
    def done(self) -> bool:
        return self.result is dcls.MISSING


class TensorRef[T: cabc.Callable = cabc.Callable]:
    """
    A data structure holding tensor information in the DAG.

    It holds the reference to the fake tensor,
    the callable that produces it, and the downstream consumer (function).

    Since in the DAG the tensor are never reused by the function,
    we can assume there is only 1 single producer.
    """

    def __init__(self, producer: T, fake: torch.Tensor):
        self._producer = producer
        self._fake = fake
        self._consumers: set[cabc.Callable] = set()

        if not isinstance(fake, torch.Tensor) or is_real(fake):
            raise ValueError(
                f"The fake tensor produced at idx={self.producer} is real."
            )

    def __hash__(self) -> int:
        return id(self.fake)

    def add_consumers(self, *consumers: cabc.Callable) -> None:
        "Add consumers for the info."

        if len(set(consumers)) != len(consumers):
            raise ValueError("Duplicate values in consumers.")

        for consumer in consumers:
            self._add_consumer(consumer)

    def _add_consumer(self, consumer: cabc.Callable) -> None:
        if consumer in self.consumers:
            raise IndexError(f"Attempting to add {consumer=} a second time.")

        self._consumers.add(consumer)

    @property
    def producer(self) -> T:
        "The producer index."
        return self._producer

    @property
    def consumers(self) -> cabc.Set[cabc.Callable]:
        "The list of consumers."
        return self._consumers

    @property
    def fake(self) -> torch.Tensor:
        "Return the fake tensor."
        return self._fake


class Dag0[F: cabc.Callable](cabc.Sequence[ThunkNode[F]]):
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(self, thunks: cabc.Iterable[ThunkNode[F]]) -> None:
        self._thunks = tuple(thunks)
        self._var_list = self._compute_local_vars()
        self._inputs = tuple(self._input_fake_vars())

    def __len__(self) -> int:
        return len(self._thunks)

    @typing.overload
    def __getitem__(self, idx: int) -> ThunkNode[F]: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Self: ...

    def __getitem__(self, idx):
        match idx:
            case slice():
                return self._getitem_slice(idx)
            case int():
                return self._getitem_int(idx)

    def _getitem_slice(self, idx: slice) -> typing.Self:
        return type(self)(self._thunks[idx])

    def _getitem_int(self, idx: int) -> ThunkNode[F]:
        return self._thunks[idx]

    def __iter__(self) -> cabc.Iterator[ThunkNode[F]]:
        return iter(self._thunks)

    def inputs(self) -> tuple[torch.Tensor, ...]:
        "The fake tensor inputs, from order or definition."

        return self._inputs

    def var_list(self) -> VarList:
        return self._var_list

    def _input_fake_vars(self) -> cabc.Generator[torch.Tensor]:
        for var in self._var_list.values():
            if self._var_list[var.fake].is_input:
                yield var.fake

    def _compute_local_vars(self) -> VarList:
        unique_vars: dict[int, VarInfo] = {}

        for idx, thunk in enumerate(self):
            self._add_inputs(idx, thunk, unique_vars)
            self._add_output(idx, thunk, unique_vars)

        return VarList(unique_vars.values())

    def _add_inputs(
        self, idx: int, thunk: ThunkNode, locals: dict[int, VarInfo]
    ) -> None:
        for input in thunk.upstreams:
            if (input_id := id(input)) not in locals:
                info = VarInfo.input_var(input)
                locals[input_id] = info

            info = locals[input_id]
            info.add_consumers(idx)

    def _add_output(
        self, idx: int, thunk: ThunkNode, locals: dict[int, VarInfo]
    ) -> None:
        # Output must be unique, so it's always new.
        for output in thunk.downstreams:
            info = VarInfo(producer=idx, fake=output)

            if id(info.fake) in locals:
                raise KeyError(f"Output produced is not unique.")

            locals[id(output)] = info


class Dag[F: cabc.Callable](cabc.Sequence[ThunkNode[F]]):
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(
        self,
        thunks: cabc.Iterable[ThunkNode[F]],
        inputs: cabc.Iterable[torch.Tensor] = (),
        outputs: cabc.Iterable[torch.Tensor] = (),
    ) -> None:
        self._thunks = tuple(thunks)

        self._inputs = frozenset(inputs)
        self._outputs = frozenset(outputs)

        self._produced_by_step = self._step_output_mapping()
        "The mapping from tensor id to step count."

        self._inputs_to_step = self._step_input_mapping()
        "The mapping from tensor id to step that uses it."

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

    def __len__(self) -> int:
        return len(self._thunks)

    def __iter__(self) -> cabc.Iterator[ThunkNode[F]]:
        return iter(self._thunks)

    def __get_all_tensors_produced(self):
        for i, thunk in enumerate(self._thunks):
            for tensor in thunk.downstreams:
                yield i, tensor

    def _step_output_mapping(self) -> dict[int, int]:
        idx_to_tensors = list(self.__get_all_tensors_produced())

        result = {}
        for i, tensor in idx_to_tensors:
            result[id(tensor)] = i

        if len(result) != len(idx_to_tensors):
            raise ValueError("Output tensors of thunks are not unique.")

        return result

    def _step_input_mapping(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = collections.defaultdict(list)

        for i, thunk in enumerate(self._thunks):
            for tensor in thunk.upstreams:
                result[id(tensor)].append(i)

        return result

    def _validate_input_output(self) -> None:
        for input in self._inputs:
            # Input should not be produced anywhere.
            if id(input) in self._produced_by_step:
                raise ValueError("Input not discovered.")

            # Input should be used.
            if id(input) not in self._inputs_to_step:
                raise ValueError("Input is not used.")

        for output in self._outputs:
            # Output is not produced.
            if id(output) not in self._produced_by_step:
                raise ValueError("Output not discovered.")
