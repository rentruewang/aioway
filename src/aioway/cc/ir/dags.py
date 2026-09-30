# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import collections
import functools
import typing
from collections import abc as cabc

import pytest
import torch

from aioway.t import TensorId, TList

from .nodes import TensorRef, ThunkNode, ThunkNodeId

__all__ = ["Dag"]


class Dag[F: cabc.Callable]:
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(
        self,
        thunks: cabc.Iterable[ThunkNode[F]],
        inputs: cabc.Iterable[torch.Tensor],
        outputs: cabc.Iterable[torch.Tensor],
    ) -> None:
        self._thunks = tuple(thunks)

        self._inputs = tuple(inputs)
        self._outputs = tuple(outputs)

        self._tensors = _build_tensor_refs(self._thunks, self._inputs)
        "Mapping from tensors to refs (linking functions)."

        self._inputs_to_step = _tensor_is_input_to_thunk(self._thunks)
        "The mapping from tensor id to thunk's id that uses it."

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self.tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __len__(self) -> int:
        return len(self._thunks)

    def __iter__(self) -> cabc.Iterator[ThunkNode[F]]:
        return iter(self._thunks)

    def __getitem__(self, idx: int):
        return self._thunks[idx]

    def first_use(self, tensor: torch.Tensor) -> int:
        return self.__get_tensor_life(tensor, min)

    def last_use(self, tensor: torch.Tensor) -> int:
        return self.__get_tensor_life(tensor, max)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @functools.cached_property
    def tensors(self) -> TList:
        return TList(self._all_tensors())

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        pytest.xfail("Fail because this should be changed to actually produce step.")
        return self._tensors[TensorId.from_tensor(tensor)].producer

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        pytest.xfail("Fail because this should be changed to actually produce step.")
        return self._inputs_to_step[id(tensor)]

    def _all_tensors(self):
        yield from self._inputs
        yield from self._outputs

        yield from _all_thunk_tensors(self._thunks)

    def _validate_input_output(self) -> None:
        for input in self._inputs:
            # Input should not have a producer.
            if not self._tensors[TensorId.from_tensor(input)].is_free:
                raise ValueError("Input is not a free variable.")

            # Input should be used.
            if id(input) not in self._inputs_to_step:
                raise ValueError("Input is not used.")

        for output in self._outputs:
            # Output is not produced.
            if id(output) not in self._tensors:
                raise ValueError("Output not discovered.")

        if set(self._inputs) & set(self._outputs):
            raise ValueError("Inputs are in the outputs. Not allowed yet.")

    def __get_tensor_life(
        self, tensor: torch.Tensor, func: cabc.Callable[[cabc.Iterable[int]], int]
    ) -> int:
        # Use the ids to check because tensor `==` compares by element.

        if id(tensor) in self._input_ids:
            return -1

        if id(tensor) in self._output_ids:
            return len(self)

        return func(self._inputs_to_step[TensorId.from_tensor(tensor)])

    @functools.cached_property
    def _input_ids(self) -> frozenset[int]:
        return frozenset(id(t) for t in self._inputs)

    @functools.cached_property
    def _output_ids(self) -> frozenset[int]:
        return frozenset(id(t) for t in self._outputs)

    @classmethod
    def from_thunk_list(cls, thunks: cabc.Sequence[ThunkNode[F]]) -> typing.Self:
        """
        Given only the thunk list, construct a DAG, auto discover inputs and outputs.

        Inputs = unproduced tensors that exists in graph.
        Outputs = unused tensors in graph.

        Both are in order of first appearance, which decides the signature.
        """

        uses = _tensor_is_input_to_thunk(thunks)
        tensors = TList(_all_thunk_tensors(thunks))

        # Output refs that is fully linked.
        outputs: dict[int, torch.Tensor] = {}
        for thunk in thunks:
            for o in thunk.outputs:
                assert id(o) not in outputs
                outputs[id(o)] = o

        input_only = uses.keys() - outputs.keys()
        output_only = outputs.keys() - uses.keys()

        # Walk `tensors` rather than the sets: set order is not insertion order.
        input_tensors = [t for t in tensors if id(t) in input_only]
        output_tensors = [t for t in tensors if id(t) in output_only]

        return cls(thunks, input_tensors, output_tensors)


def _all_thunk_tensors(thunks: cabc.Sequence[ThunkNode]):
    for thunk in thunks:
        yield from thunk.inputs
        yield from thunk.outputs


def _tensor_is_input_to_thunk(
    thunks: cabc.Sequence[ThunkNode],
) -> dict[TensorId, list[ThunkNode]]:
    result: dict[TensorId, list[ThunkNode]] = collections.defaultdict(list)

    for thunk in thunks:
        for tensor in thunk.inputs:
            result[TensorId.from_tensor(tensor)].append(thunk)

    return result


def _build_tensor_refs(
    thunks: cabc.Sequence[ThunkNode], inputs: cabc.Sequence[torch.Tensor]
) -> dict[TensorId, TensorRef]:
    "Get the mapping from id of `torch.Tensor` to corresponding tensor ref."

    # Register all inputs.
    mapping: dict[TensorId, TensorRef] = _tensor_refs_inputs(inputs)

    # Register all outputs of thunks.
    mapping |= _tensor_refs_outputs(thunks)

    # Register each thunk's input.
    _link_inputs_for_mapping(mapping, thunks)

    return mapping


def _tensor_refs_inputs(
    inputs: cabc.Sequence[torch.Tensor],
) -> dict[TensorId, TensorRef]:
    mapping: dict[TensorId, TensorRef] = {}

    for input in inputs:
        ref = TensorRef(producer=None, tensor=input)
        mapping[ref.__tensor__id__] = ref

    return mapping


def _tensor_refs_outputs(thunks: cabc.Sequence[ThunkNode]) -> dict[TensorId, TensorRef]:
    mapping: dict[TensorId, TensorRef] = {}

    for thunk in thunks:
        for output in thunk.outputs:
            if id(output) in mapping:
                raise ValueError("The output fake tensor is not unique.")

            ref = TensorRef(producer=thunk.__id__, tensor=output)
            mapping[ref.__tensor__id__] = ref
    return mapping


def _link_inputs_for_mapping(
    mapping: dict[TensorId, TensorRef], thunks: cabc.Sequence[ThunkNode]
) -> None:
    for thunk in thunks:
        for input in thunk.inputs:
            assert id(input) in mapping
            mapping[TensorId.from_tensor(input)].add_consumers(thunk.__id__)
