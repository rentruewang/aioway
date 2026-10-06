# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import copy
import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import AnyDict, AnySet, IntArray, any_dict, any_set
from aioway.ir.instrs import Instr, InstrList
from aioway.t import TList, all_real, parse_attr

if typing.TYPE_CHECKING:
    from aioway.ir import Query

__all__ = ["TensorRef", "Program", "TensorLifetime"]

# The DAG class ====


class Program[I: Instr = typing.Any]:
    """
    A program is a DAG of callables, that are linked by fake tensors.
    """

    def __init__(
        self,
        instrs: cabc.Iterable[I],
        inputs: cabc.Iterable[torch.Tensor],
        outputs: cabc.Iterable[torch.Tensor],
    ) -> None:
        self._instrs = InstrList(instrs)

        self._inputs = TList.from_self_or_iter(inputs)
        self._outputs = TList.from_self_or_iter(outputs)

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self.tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __repr__(self) -> str:
        return repr(self.instrs)

    def __len__(self) -> int:
        return len(self.instrs)

    def __iter__(self) -> cabc.Iterator[I]:
        return iter(self.instrs)

    @typing.overload
    def __getitem__(self, idx: Query) -> typing.Self: ...

    @typing.overload
    def __getitem__(self, idx: int) -> I: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int] | IntArray) -> list[I]: ...

    def __getitem__(self, idx):
        from aioway.ir import Query

        if isinstance(idx, int):
            return self.instrs[idx]

        if isinstance(idx, Query):
            return idx.select(self)

        # Convert `slice` to `list[int]` with help of `range`.
        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])

        # If it's `list[int]` or `IntArray`.
        if np.isdtype((arr := np.asarray(idx)).dtype, "integral"):
            return [self.instrs[i] for i in arr]

        raise TypeError(f"Unknown type: {type(idx)=}.")

    def __setitem__(self, q: Query, subset: Program) -> None:
        mutated = _replace_subset(prog=self, query=q, subset=subset)

        # Overwrite the references s.t. underlying data is not touched.
        # This is not supposed to fail because we already constructed a program.
        self.__init__(
            instrs=mutated.instrs, inputs=mutated.inputs, outputs=mutated.outputs
        )

    @functools.cached_property
    def _tensor_links(self) -> AnyDict[torch.Tensor, TensorRef]:
        "Mapping from tensors to refs (linking functions)."
        return _build_tensor_refs(self.instrs, list(self._inputs))

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
    def instrs(self) -> InstrList[I]:
        return self._instrs

    @property
    def inputs(self) -> TList:
        return self._inputs

    @property
    def outputs(self) -> TList:
        return self._outputs

    @functools.cached_property
    def tensors(self) -> TList:
        return TList.from_iterable(self._all_tensors())

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        ins = self._instr_producing(tensor)
        return self.instrs.index(ins) if ins is not None else -1

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        consuming = self._thunk_consuming(tensor)
        return [self.instrs.index(thunk) for thunk in consuming]

    def _all_tensors(self):
        yield from self._inputs
        yield from self._outputs

        yield from _all_thunk_tensors(self.instrs)

    def _validate_input_output(self) -> None:
        for input in self._inputs:
            # Input should not have a producer.
            if not self._tensor_links[input].is_free:
                raise ValueError("Input is not a free variable.")

        for output in self._outputs:
            # Output is not produced.
            if output not in self._tensor_links:
                raise ValueError("Output not discovered.")

        if set(self._inputs) & set(self._outputs):
            raise ValueError("Inputs are in the outputs. Not allowed yet.")

    def _instr_producing(self, tensor: torch.Tensor, /) -> I | None:
        """
        Get the thunk that produces the producer.

        If it's from the input, return `None`.
        """

        if tensor in self.inputs:
            return None

        else:
            return self._tensor_links[tensor].producer

    def _thunk_consuming(self, tensor: torch.Tensor, /) -> cabc.Generator[I]:
        ref = self._tensor_links[tensor]
        yield from ref.consumers

    def copy(self) -> typing.Self:
        "Do a shallow copy of `self`, for CoW."
        return copy.copy(self)

    @classmethod
    def from_thunk_list(cls, thunks: cabc.Iterable[I]) -> typing.Self:
        """
        Given only the thunk list, construct a DAG, auto discover inputs and outputs.

        Inputs = unproduced tensors that exists in graph.
        Outputs = unused tensors in graph.

        Both are in order of first appearance, which decides the signature.
        """

        uses = _tensor_is_input_to_thunk(thunks)
        tensors = TList.from_iterable(_all_thunk_tensors(thunks))

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


class TensorRef[F: Instr]:
    """
    A data structure holding tensor information in the DAG.

    It holds the reference to the fake tensor,
    the callable that produces it, and the downstream consumer (function).

    Since in the DAG the tensor are never reused by the function,
    we can assume there is only 1 single producer.
    """

    def __init__(self, producer: F | None, tensor: torch.Tensor) -> None:
        """
        Args:
            producer: The thunk's id, or `None` if it's a free variable.
            tensor: The tensor this reference tracks. Must be fake.
        """

        self._producer = producer
        self._tensor = tensor
        self._consumers: AnySet[F] = typing.cast(typing.Any, any_set(F))

        if not isinstance(tensor, torch.Tensor) or all_real(tensor):
            raise ValueError(
                f"The fake tensor produced at idx={self._producer} is real."
            )

    def __repr__(self) -> str:
        attr = parse_attr(self.tensor)
        return f"TRef({attr!s}, {self.producer})"

    def add_consumers(self, *consumers: F) -> None:
        "Add consumers for the info. Allow duplication."

        for consumer in consumers:
            self._consumers.add(consumer)

    @property
    def producer(self) -> F:
        "The producer index."

        if self._producer is None:
            raise AttributeError("The variable is a free variable.")
        else:
            return self._producer

    @property
    def consumers(self) -> AnySet[F]:
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


def _all_thunk_tensors(thunks: cabc.Iterable[Instr]) -> cabc.Generator[torch.Tensor]:
    for thunk in thunks:
        yield from thunk.inputs
        yield from thunk.outputs


def _tensor_is_input_to_thunk[T: Instr](
    thunks: cabc.Iterable[T],
) -> AnyDict[torch.Tensor, list[T]]:
    result: AnyDict[torch.Tensor, list[T]] = any_dict(torch.Tensor)

    for thunk in thunks:
        for tensor in thunk.inputs:
            if tensor not in result:
                result[tensor] = []

            result[tensor].append(thunk)

    return result


def _build_tensor_refs(
    thunks: cabc.Iterable[Instr], inputs: cabc.Sequence[torch.Tensor]
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


def _tensor_refs_outputs[I: Instr](
    thunks: cabc.Iterable[I],
) -> AnyDict[torch.Tensor, TensorRef[I]]:
    mapping: AnyDict[torch.Tensor, TensorRef[I]] = any_dict(torch.Tensor)

    for thunk in thunks:
        for output in thunk.outputs:
            if output in mapping:
                raise ValueError("The output fake tensor is not unique.")

            ref = TensorRef(producer=thunk, tensor=output)
            mapping[ref.tensor] = ref
    return mapping


def _link_inputs_for_mapping(
    mapping: AnyDict[torch.Tensor, TensorRef], thunks: cabc.Iterable[Instr]
) -> None:
    for thunk in thunks:
        for input in thunk.inputs:
            assert input in mapping
            mapping[input].add_consumers(thunk)


def _replace_subset[I: Instr](
    *, prog: Program[I], query: Query, subset: Program[I]
) -> Program[I]:
    """
    Replace the query with a new subset.
    """

    queried = query.select(prog)

    if queried.inputs.attrs() != subset.inputs.attrs():
        raise ValueError("Inputs are not compatible.")

    if queried.outputs.attrs() != subset.outputs.attrs():
        raise ValueError("Outputs are not compatible.")

    # Get the indices of the queried subnet and minimum (useful in inserting).
    qidx = {prog.instrs.index(q) for q in queried}
    min_qidx = min(qidx)

    # Inputs and outputs are not shared.
    assert subset.inputs.keys().isdisjoint(subset.outputs.keys())

    # Build mapping for replacement.
    in_to_out = any_dict(torch.Tensor)
    for before, after in zip(queried.inputs, subset.inputs):
        in_to_out[before] = after
    for before, after in zip(queried.outputs, subset.outputs):
        in_to_out[before] = after

    # Drop the ones that are queried.
    new_instrs: list[I] = [
        thunk.tree_map_only(torch.Tensor, lambda t: in_to_out.get(t, t))
        for i, thunk in enumerate(prog.instrs)
        if i not in qidx
    ]

    # Replace with new.
    pre = new_instrs[:min_qidx]
    post = new_instrs[min_qidx:]
    return prog.from_thunk_list([*pre, *subset.instrs, *post])


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
