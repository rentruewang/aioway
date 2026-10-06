# Copyright (c) AIoWay Authors - All Rights Reserved

"The DAG that supports analysis."

import copy
import dataclasses as dcls
import functools
import operator
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import AnyDict, AnySet, IntArray, any_dict, any_set, is_list_of
from aioway.ir.instrs import Instr, InstrList
from aioway.t import TList, all_real, parse_attr

__all__ = ["TensorRef", "Program", "Query", "TensorLifetime"]

# The DAG class ====


@typing.runtime_checkable
class Query[I: Instr = typing.Any](typing.Protocol):
    def __call__(self, program: Program[I], /) -> list[int]: ...


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
        self._instrs = InstrList.build(instrs)

        self._inputs = TList.from_self_or_iter(inputs)
        self._outputs = TList.from_self_or_iter(outputs)

        self._tensors = TList.from_iterable(self._all_tensors())
        self._tensor_links = _build_tensor_refs(self.instrs, list(self._inputs))

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
    def __getitem__(self, idx: int) -> I: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[int] | IntArray | Query) -> typing.Self: ...

    def __getitem__(self, idx):
        # For numpy objects, convert to `int | list[int]`.
        if isinstance(idx, np.ndarray | np.generic):
            # If it's `IntArray`.
            if np.isdtype(idx.dtype, "integral") and idx.ndim in [0, 1]:
                idx = idx.tolist()
            else:
                raise IndexError("Only 0D or 1D numpy array supported.")

        if isinstance(idx, int):
            return self.instrs[idx]

        if isinstance(idx, Query):
            idx = idx(self)

        # Convert `slice` to `list[int]` with help of `range`.
        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])

        # Finally, handle `list[int]`.
        if is_list_of(int)(idx):
            sel, inputs, outputs = _select_indices(indices=idx, prog=self)
            return type(self)(instrs=sel, inputs=inputs, outputs=outputs)

        raise TypeError(f"Unknown type: {type(idx)=}.")

    def __setitem__(self, query: Query, subset: Program) -> None:
        mutated = _replace_sub_prog(prog=self, query=query, sub_prog=subset)

        # Overwrite the references s.t. underlying data is not touched.
        # This is not supposed to fail.
        self.__init__(
            instrs=mutated.ilist, inputs=mutated.inputs, outputs=mutated.outputs
        )

    @property
    def tensor_links(self) -> AnyDict[torch.Tensor, TensorRef]:
        "Mapping from tensors to refs (linking functions)."
        return self._tensor_links

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

    @property
    def tensors(self) -> TList:
        "The tensors that exist in this program."
        return self._tensors

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
    def from_instr_list(cls, instrs: cabc.Iterable[I], /) -> typing.Self:
        """
        Given only the thunk list, construct a DAG, auto discover inputs and outputs.

        Inputs = unproduced tensors that exists in graph.
        Outputs = unused tensors in graph.

        Both are in order of first appearance, which decides the signature.
        """

        instrs = tuple(instrs)
        io_only = _InputOutputOnly.from_instrs(instrs)

        return cls(instrs, inputs=io_only.inputs, outputs=io_only.outputs)


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


def _used_outside(prog: Program, selected_idx: set[int]) -> TList:
    """
    Add all tensors used outside of selected region.
    """

    used = TList.empty()

    # Only check those that occur after, as this is a DAG.
    for i in range(min(selected_idx) + 1, len(prog)):
        if i in selected_idx:
            continue

        used += prog[i].inputs

    # Populate the outputs as it's considered "consumed".
    used += prog.outputs

    return used


# The querying related utilities ====


class _ProgramTuple[I: Instr](typing.NamedTuple):
    "A namedtuple to avoid paying construction cost of `Program`."

    ilist: InstrList[I]
    inputs: TList
    outputs: TList


def _replace_sub_prog[I: Instr](
    *, prog: Program[I], query: Query[I], sub_prog: Program[I]
) -> _ProgramTuple[I]:
    """
    Replace the query with a new subset.
    """

    queried = _query_select(query=query, prog=prog)

    if queried.inputs.attrs() != sub_prog.inputs.attrs():
        raise ValueError("Inputs are not compatible.")

    if queried.outputs.attrs() != sub_prog.outputs.attrs():
        raise ValueError("Outputs are not compatible.")

    # Get the indices of the queried subnet and minimum (useful in inserting).
    qidx = {prog.instrs.index(q) for q in queried.ilist}
    min_qidx = min(qidx)

    # Inputs and outputs are not shared.
    assert sub_prog.inputs.keys().isdisjoint(sub_prog.outputs.keys())

    # Build mapping for replacement.
    in_to_out = any_dict(torch.Tensor)
    for before, after in zip(queried.inputs, sub_prog.inputs):
        in_to_out[before] = after
    for before, after in zip(queried.outputs, sub_prog.outputs):
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
    replaced = [*pre, *sub_prog.instrs, *post]
    io_only = _InputOutputOnly.from_instrs(replaced)

    return _ProgramTuple(
        ilist=InstrList.build(replaced),
        inputs=io_only.inputs,
        outputs=io_only.outputs,
    )


def _query_select[I: Instr](*, query: Query[I], prog: Program[I]) -> _ProgramTuple[I]:
    # Sorted and deduplicated, so `selected` is in step order.
    indices = query(prog)
    return _select_indices(indices=indices, prog=prog)


def _select_indices[I: Instr](
    *, indices: list[int], prog: Program[I]
) -> _ProgramTuple[I]:
    """
    Produce a subprogram whose:
    Input is any tensor used in this scope but not defined in the scope.
    Output is any tensor produced and used in downstream.
    """

    if indices != sorted(indices):
        raise ValueError("The index provided should be sorted and unique.")

    idx_set = set(indices)
    selected = prog.instrs[indices]

    # Using `TList` as it has fast lookups and ergonomic operations.
    produced_here = functools.reduce(operator.add, (i.outputs for i in selected))
    used_outside = _used_outside(prog, idx_set)

    # Inputs: used by our selected by not produced inside the region.
    inputs = TList.from_iterable(
        t for i in selected for t in i.inputs if t not in produced_here
    )

    # Outputs: produced by our selected and used by outside thunks.
    outputs = TList.from_iterable(
        t for i in selected for t in i.outputs if t in used_outside
    )

    # Inputs should not depend on intermediate.
    # Since `inputs_produced_by` represent intermediate, it can be empty.
    inputs_produced_by = [prog.output_of_step(t) for t in inputs]
    if inputs_produced_by and min(indices) < max(inputs_produced_by):
        raise ValueError("Illegal subset where input depend on intermediate.")

    return _ProgramTuple(selected, inputs, outputs)


# Helper classes ====


@dcls.dataclass(frozen=True)
class _InputOutputOnly:
    inputs: TList
    outputs: TList

    @classmethod
    def from_instrs[I: Instr](cls, instrs: cabc.Sequence[I]) -> typing.Self:
        # Inputs observed in the list of thunks.
        tensor_in_list = TList.empty()

        # Output observed in the list of thunks.
        tensor_out_list = TList.empty()

        for thunk in instrs:
            tensor_in_list += thunk.inputs

            # Output must be unique.
            assert tensor_out_list.isdisjoint(thunk.outputs)
            tensor_out_list += thunk.outputs

        input_only = tensor_in_list - tensor_out_list
        output_only = tensor_out_list - tensor_in_list

        return cls(inputs=input_only, outputs=output_only)


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
