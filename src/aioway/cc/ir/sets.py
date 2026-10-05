# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import AnyDict, AnySet, IntArray, any_dict, any_set
from aioway.t import TList, fake_mode, is_real, parse_attr

from .instrs import FCall

if typing.TYPE_CHECKING:
    from .queries import Query

__all__ = ["TensorRef", "InstrSet", "TensorLifetime"]

# The DAG class ====


class InstrSet[T: FCall]:
    """
    A dag is a sequence of callables, that are linked by fake tensors.
    """

    def __init__(
        self,
        thunks: cabc.Iterable[T],
        inputs: cabc.Iterable[torch.Tensor],
        outputs: cabc.Iterable[torch.Tensor],
    ) -> None:
        self._thunks = tuple(thunks)

        self._inputs = TList(inputs)
        self._outputs = TList(outputs)

        self._thunk_to_step: AnyDict[T, int] = any_dict(
            FCall, *((thunk, idx) for idx, thunk in enumerate(self._thunks))
        )
        "Mapping from thunks to their indices."

        self._tensor_links = _build_tensor_refs(self._thunks, list(self._inputs))
        "Mapping from tensors to refs (linking functions)."

        self._inputs_to_thunk_index = _tensor_is_input_to_thunk(self._thunks)
        "The mapping from tensor to thunk's that uses it."

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self.tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __repr__(self) -> str:
        return repr(list(self._thunks))

    def __len__(self) -> int:
        return len(self._thunks)

    def __iter__(self) -> cabc.Iterator[T]:
        return iter(self._thunks)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: list[int] | IntArray) -> list[FCall]: ...

    @typing.overload
    def __getitem__(self, idx: Query) -> typing.Self: ...

    def __getitem__(self, idx):
        from .queries import Query

        if isinstance(idx, int):
            return self._thunks[idx]

        if isinstance(idx, Query):
            return idx(self)

        # If it's `list[int]` or `IntArray`.
        if np.isdtype((arr := np.asarray(idx)).dtype, "integral"):
            return [self._thunks[i] for i in arr]

        raise TypeError(f"Unknown type: {type(idx)=}.")

    def __setitem__(self, query: Query, subset: typing.Self) -> None:
        queried = self[query]

        if queried.inputs.attrs() != subset.inputs.attrs():
            raise ValueError("Inputs are not compatible.")

        if queried.outputs.attrs() != subset.outputs.attrs():
            raise ValueError("Outputs are not compatible.")

        last_sub_step = max(self.index(q) for q in queried)

        with fake_mode():
            # Retrace every step after the maximum index.
            x = queried

    def parents(self, thunk: T) -> AnySet[T]:
        return any_set(FCall, *self._parents(thunk))

    def children(self, thunk: T) -> AnySet[T]:
        return any_set(FCall, *self._children(thunk))

    def _parents(self, thunk: T) -> cabc.Generator[T]:
        for input in thunk.inputs:
            ref = self._tensor_links[input]

            # Check if it is a free variable.
            if not ref.is_free:
                yield ref.producer

    def _children(self, thunk: T) -> cabc.Generator[T]:
        for output in thunk.outputs:
            ref = self._tensor_links[output]
            yield from ref.consumers

    def life(self, t: torch.Tensor, /) -> TensorLifetime:
        """
        Compute the lifetime of a tensor.

        birth = at which point we start keeping track of it.
        death = at which point it is no longer needed.
        """

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

    def index(self, instr: T, /) -> int:
        "Get the index of each instruction."
        return self._thunk_to_step[instr]

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        thunk = self._thunk_producing(tensor)
        return self.index(thunk) if thunk is not None else -1

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        consuming = self._thunk_consuming(tensor)
        return [self.index(thunk) for thunk in consuming]

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

    def _thunk_producing(self, tensor: torch.Tensor, /) -> T | None:
        """
        Get the thunk that produces the producer.

        If it's from the input, return `None`.
        """

        if tensor in self.inputs:
            return None

        else:
            return self._tensor_links[tensor].producer

    def _thunk_consuming(self, tensor: torch.Tensor, /) -> cabc.Generator[T]:
        ref = self._tensor_links[tensor]
        yield from ref.consumers

    @classmethod
    def from_thunk_list(cls, thunks: cabc.Sequence[T]) -> typing.Self:
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


# The node classes ====


class TensorRef[T: FCall]:
    """
    A data structure holding tensor information in the DAG.

    It holds the reference to the fake tensor,
    the callable that produces it, and the downstream consumer (function).

    Since in the DAG the tensor are never reused by the function,
    we can assume there is only 1 single producer.
    """

    def __init__(self, producer: T | None, tensor: torch.Tensor) -> None:
        """
        Args:
            producer: The thunk's id, or `None` if it's a free variable.
            tensor: The tensor this reference tracks. Must be fake.
        """

        self._producer = producer
        self._tensor = tensor
        self._consumers: AnySet[T] = typing.cast(typing.Any, any_set(FCall))

        if not isinstance(tensor, torch.Tensor) or is_real(tensor):
            raise ValueError(
                f"The fake tensor produced at idx={self._producer} is real."
            )

    def __repr__(self) -> str:
        attr = parse_attr(self.tensor)
        return f"TRef({attr!s}, {self.producer})"

    def add_consumers(self, *consumers: T) -> None:
        "Add consumers for the info. Allow duplication."

        for consumer in consumers:
            self._consumers.add(consumer)

    @property
    def producer(self) -> T:
        "The producer index."

        if self._producer is None:
            raise AttributeError("The variable is a free variable.")
        else:
            return self._producer

    @property
    def consumers(self) -> AnySet[T]:
        "The list of consumers."
        return self._consumers

    @property
    def tensor(self) -> torch.Tensor:
        "Return the fake tensor."
        return self._tensor

    @property
    def is_free(self) -> bool:
        "Check if the tensor is a free value."
        return self._producer is None


# Helper functions for dag ====


def _all_thunk_tensors(thunks: cabc.Sequence[FCall]):
    for thunk in thunks:
        yield from thunk.inputs
        yield from thunk.outputs


def _tensor_is_input_to_thunk(
    thunks: cabc.Sequence[FCall],
) -> AnyDict[torch.Tensor, list[FCall]]:
    result: AnyDict[torch.Tensor, list[FCall]] = any_dict(torch.Tensor)

    for thunk in thunks:
        for tensor in thunk.inputs:
            if tensor not in result:
                result[tensor] = []

            result[tensor].append(thunk)

    return result


def _build_tensor_refs(
    thunks: cabc.Sequence[FCall], inputs: cabc.Sequence[torch.Tensor]
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
    thunks: cabc.Sequence[FCall],
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
    mapping: AnyDict[torch.Tensor, TensorRef], thunks: cabc.Sequence[FCall]
) -> None:
    for thunk in thunks:
        for input in thunk.inputs:
            assert input in mapping
            mapping[input].add_consumers(thunk)


# Helper classes ====


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
