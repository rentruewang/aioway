# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import torch

from aioway._utils import AnyDict, AnySet, any_dict, any_set
from aioway.t import TList

from .nodes import TensorRef, ThunkNode

__all__ = ["Dag", "TensorLifetime"]


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

        self._inputs = TList(inputs)
        self._outputs = TList(outputs)

        self._thunk_to_step: AnyDict[ThunkNode[F], int] = any_dict(
            ThunkNode, *((thunk, idx) for idx, thunk in enumerate(self._thunks))
        )

        self._tensor_links = _build_tensor_refs(self._thunks, list(self._inputs))
        "Mapping from tensors to refs (linking functions)."

        self._inputs_to_thunk_index = _tensor_is_input_to_thunk(self._thunks)
        "The mapping from tensor id to thunk's id that uses it."

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self.tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __len__(self) -> int:
        return len(self._thunks)

    def __iter__(self) -> cabc.Iterator[ThunkNode[F]]:
        return iter(self._thunks)

    def __getitem__(self, idx: int) -> ThunkNode[F]:
        return self._thunks[idx]

    def life(self, t: torch.Tensor, /) -> TensorLifetime:
        birth = self.output_of_step(t) if t not in self.inputs else -1
        death = max(self.input_to_step(t)) if t not in self.outputs else len(self)
        return TensorLifetime(birth=birth, death=death)

    @property
    def inputs(self) -> TList:
        return self._inputs

    @property
    def outputs(self) -> TList:
        return self._outputs

    @functools.cached_property
    def tensors(self) -> TList:
        return TList(self._all_tensors())

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        thunk = self._tensor_links[tensor].producer
        return self._thunk_to_step[thunk]

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:

        thunks: AnySet[ThunkNode[F]] = self._tensor_links[tensor].consumers
        return [self._thunk_to_step[thunk] for thunk in thunks]

    def _all_tensors(self):
        yield from self._inputs
        yield from self._outputs

        yield from _all_thunk_tensors(self._thunks)

    def _validate_input_output(self) -> None:
        for input in self._inputs:
            # Input should not have a producer.
            if not self._tensor_links[input].is_free:
                raise ValueError("Input is not a free variable.")

            # Input should be used.
            if input not in self._inputs_to_thunk_index:
                raise ValueError("Input is not used.")

        for output in self._outputs:
            # Output is not produced.
            if output not in self._tensor_links:
                raise ValueError("Output not discovered.")

        if set(self._inputs) & set(self._outputs):
            raise ValueError("Inputs are in the outputs. Not allowed yet.")

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

        # Output observed in the list of thunks.
        tensor_out_list = any_set(torch.Tensor)
        for thunk in thunks:
            for o in thunk.outputs:
                assert o not in tensor_out_list
                tensor_out_list.add(o)

        input_only = uses.keys() - tensor_out_list
        output_only = tensor_out_list - uses.keys()

        # Walk `tensors` rather than the sets: set order is not insertion order.
        input_tensors = [t for t in tensors if t in input_only]
        output_tensors = [t for t in tensors if t in output_only]

        return cls(thunks, input_tensors, output_tensors)


# Helper functions ----


def _all_thunk_tensors(thunks: cabc.Sequence[ThunkNode]):
    for thunk in thunks:
        yield from thunk.inputs
        yield from thunk.outputs


def _tensor_is_input_to_thunk(
    thunks: cabc.Sequence[ThunkNode],
) -> AnyDict[torch.Tensor, list[ThunkNode]]:
    result: AnyDict[torch.Tensor, list[ThunkNode]] = any_dict(torch.Tensor)

    for thunk in thunks:
        for tensor in thunk.inputs:
            if tensor not in result:
                result[tensor] = []

            result[tensor].append(thunk)

    return result


def _build_tensor_refs(
    thunks: cabc.Sequence[ThunkNode], inputs: cabc.Sequence[torch.Tensor]
) -> AnyDict[torch.Tensor, TensorRef]:
    "Get the mapping from id of `torch.Tensor` to corresponding tensor ref."

    # Register all inputs.
    mapping: AnyDict[torch.Tensor, TensorRef] = _tensor_refs_inputs(inputs)

    # Register all outputs of thunks.
    mapping |= _tensor_refs_outputs(thunks)

    # Register each thunk's input.
    _link_inputs_for_mapping(mapping, thunks)

    return mapping


def _tensor_refs_inputs(
    inputs: cabc.Sequence[torch.Tensor],
) -> AnyDict[torch.Tensor, TensorRef]:
    mapping: AnyDict[torch.Tensor, TensorRef] = any_dict(torch.Tensor)

    for input in inputs:
        ref = TensorRef(producer=None, tensor=input)
        mapping[ref.tensor] = ref

    return mapping


def _tensor_refs_outputs(
    thunks: cabc.Sequence[ThunkNode],
) -> AnyDict[torch.Tensor, TensorRef]:
    mapping: AnyDict[torch.Tensor, TensorRef] = any_dict(torch.Tensor)

    for thunk in thunks:
        for output in thunk.outputs:
            if output in mapping:
                raise ValueError("The output fake tensor is not unique.")

            ref = TensorRef(producer=thunk, tensor=output)
            mapping[ref.tensor] = ref
    return mapping


def _link_inputs_for_mapping(
    mapping: AnyDict[torch.Tensor, TensorRef], thunks: cabc.Sequence[ThunkNode]
) -> None:
    for thunk in thunks:
        for input in thunk.inputs:
            assert input in mapping
            mapping[input].add_consumers(thunk)


# Helper classes ----


@dcls.dataclass(frozen=True)
class TensorLifetime:
    "Report the steps that the tensor is produced and last used."

    birth: int
    "The brith time. Inputs are -1, others are all positive."

    death: int
    "The death time. Outputs are `len(dag)`, of course it depends on which DAG."

    def __post_init__(self) -> None:
        if self.birth < -1:
            raise ValueError("Only -1 or positive numbers.")

        if self.death < self.birth:
            raise ValueError("Death before birth for lifetime.")
