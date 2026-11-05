# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Proc` that supports analysis."

import contextlib as ctxl
import copy
import dataclasses as dcls
import functools
import operator
import textwrap
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import AnySet, IntArray, any_set, is_list_of
from aioway.ir.instrs import Instr, InstrList
from aioway.ir.queries import OrderedIndex, Query
from aioway.t import (
    BoundArgs,
    KeyEntry,
    MappingKey,
    TList,
    all_real,
    is_tensor_container,
    parse_attr,
    tree_leaves,
    tree_map_only,
)

__all__ = ["Proc", "TensorRef", "TensorIndex", "TensorLifetime"]

# The Proc class ====

_LAMBDA = "<lambda>"


class Proc[I: Instr = typing.Any]:
    """
    A `Proc` is a "procedure".

    In the simplest form, it's a DAG (list of topo sorted nodes),
    and edges (nodes referencing the same fake tensor).

    The `Proc` owns the instructions and its signature (inputs and outputs).
    Tensor-level bookkeeping (producers, consumers, lifetimes) lives in
    `TensorIndex`, available as `Proc.tindex`.
    """

    INSTR: typing.ClassVar[type[Instr]] = Instr
    "The instr base class. If not specified, anything is ok."

    def __init__(
        self,
        *,
        instrs: cabc.Iterable[I],
        inputs: BoundArgs,
        outputs: typing.Any,
    ) -> None:
        if not isinstance(inputs, BoundArgs):
            raise TypeError(f"Inputs should be a `BoundArgs`. Got {type(inputs)=}.")

        if not is_tensor_container(outputs):
            raise TypeError(
                f"Outputs must be a container of tensors. Got {type(outputs)=}."
            )

        self._instrs = InstrList.build(instrs)

        self._inputs = inputs
        self._outputs = outputs

        # Set the deafult name to "<lambda>".
        self.__name__ = _LAMBDA

        # Link producers / consumers, and validate inputs and outputs.
        self.reset_tindex()

        if (base := self.instrs.base) and not issubclass(base, self.INSTR):
            raise TypeError(
                f"The base of instr: {self.instrs.base} is not {self.INSTR}"
            )

    def __repr__(self) -> str:
        return textwrap.dedent(
            f"""
        inputs: {self.inputs!r}
        outputs: {self.outputs!r}
        instrs: {self.instrs!r}
        """
        ).strip()

    def __str__(self) -> str:
        return str(self._code)

    def __rich__(self):
        return self._code

    def __len__(self) -> int:
        return len(self.instrs)

    def __iter__(self) -> cabc.Iterator[I]:
        return iter(self.instrs)

    @typing.overload
    def __getitem__(self, idx: int) -> I: ...

    @typing.overload
    def __getitem__(
        self, idx: slice | list[int] | IntArray | Query | OrderedIndex
    ) -> typing.Self: ...

    def __getitem__(self, idx):
        # For numpy objects, convert to `int | list[int]` via `.tolist()`.
        if isinstance(idx, np.ndarray | np.generic):
            # If it's `IntArray`.
            if np.isdtype(idx.dtype, "integral") and idx.ndim in [0, 1]:
                idx = idx.tolist()
            else:
                raise IndexError("Only 0D or 1D numpy array supported.")

        if isinstance(idx, int):
            return self.instrs[idx]

        # Convert `slice` to `list[int]` with help of `range`.
        if isinstance(idx, slice):
            idx = list(range(len(self))[idx])

        if is_list_of(int)(idx):
            idx = OrderedIndex.from_list_int(idx)

        # Do the query.
        if isinstance(idx, Query):
            idx = idx(self)

        # Should all be `QuerySel` now.
        if not isinstance(idx, OrderedIndex):
            raise TypeError(f"Unknown index type: {type(idx)=}.")

        sel, inputs, outputs = _select_indices(selection=idx, prog=self)
        return type(self)(instrs=sel, inputs=inputs, outputs=outputs)

    def __setitem__(self, query: Query, subset: typing.Self) -> None:
        mutated = _replace_sub_proc(proc=self, query=query, sub_proc=subset)

        # Overwrite the references s.t. underlying data is not touched.
        # This is not supposed to fail. Also rebuilds the tensor index.
        self.__init__(
            instrs=mutated.ilist, inputs=mutated.inputs, outputs=mutated.outputs
        )

    @ctxl.contextmanager
    def rename(self, name: str) -> cabc.Generator[None]:
        """
        Temporarily rename the program to something else.
        """

        orig, self.__name__ = self.__name__, name
        try:
            yield
        finally:
            self.__name__ = orig

    @property
    def _code(self):
        from aioway.ir import render_func_proc

        return render_func_proc(self)

    @property
    def instrs(self) -> InstrList[I]:
        return self._instrs

    @property
    def inputs(self) -> BoundArgs:
        "The inputs, bound to the procedure's signature. Flat tensors: `.tensors`."
        return self._inputs

    @property
    def outputs(self) -> typing.Any:
        return self._outputs

    @property
    def tindex(self) -> TensorIndex[I]:
        "The tensor-level index: producers, consumers and lifetimes."
        return self._tindex

    def reset_tindex(self) -> None:
        "(Re)build the tensor index from the current instrs, inputs and outputs."
        self._tindex = TensorIndex(self)

    def copy(self) -> typing.Self:
        "Do a shallow copy of `self`, for CoW."
        copied = copy.copy(self)
        copied.reset_tindex()
        return copied

    @classmethod
    def from_instr_list(cls, instrs: cabc.Iterable[I], /) -> typing.Self:
        """
        Given only the thunk list, construct a DAG, auto discover inputs and outputs.

        Inputs = unproduced tensors that exists in graph.
        Outputs = unused tensors in graph.

        Both are in order of first appearance, which decides the signature.
        Inputs are bound positionally, to `t0`, `t1`, ...
        """

        instrs = tuple(instrs)

        consumed = TList()
        produced = TList()

        for instr in instrs:
            consumed |= instr.tensor_inputs

            assert produced.isdisjoint(outs := instr.tensor_outputs)
            produced |= outs

        return cls(
            instrs=instrs,
            inputs=BoundArgs.from_iters(consumed - produced),
            outputs=produced - consumed,
        )


def _map_key(name: str) -> KeyEntry:
    return typing.cast(typing.Any, MappingKey)(name)


# The TensorIndex class ====


class TensorIndex[I: Instr = typing.Any]:
    """
    Tensor-level index over a `Proc`,
    to decouple input output from lifetimes.

    Maps every tensor to a `TensorRef` (its producer and consumers),
    and provide info step and lifetime queries about tensors.

    `Proc` builds this index on construction.
    """

    def __init__(self, procedure: Proc[I], /) -> None:
        self._proc = procedure

        self._tensors = TList.deduplicate(self._all_tensors())
        self._links = _build_tensor_refs(
            procedure.instrs, list(procedure.inputs.tensors)
        )

        # Validate if the inputs and outputs are valid.
        self._validate_input_output()

        if not self._tensors.all_fake:
            raise ValueError("Contains non fake tensors.")

    def __contains__(self, tensor: torch.Tensor) -> bool:
        return tensor in self._links

    def __getitem__(self, tensor: torch.Tensor) -> TensorRef[I]:
        "The ref (producer and consumers) of `tensor`."
        return self._links[tensor]

    @property
    def proc(self) -> Proc[I]:
        "The procedure being indexed."
        return self._proc

    @property
    def tensors(self) -> TList:
        "The tensors that exist in the instructions, inputs and outputs."
        return self._tensors

    @property
    def links(self) -> dict[torch.Tensor, TensorRef[I]]:
        "Mapping from tensors to refs (linking functions)."
        return self._links

    def life(self, t: torch.Tensor, /) -> TensorLifetime:
        """
        Compute the lifetime of a tensor.

        birth = at which point we start keeping track of it.
        death = at which point it is no longer needed.
        """

        proc = self._proc
        birth = self.output_of_step(t) if t not in proc.inputs.tensors else -1
        death = max(self.input_to_step(t)) if t not in proc.outputs else len(proc)
        return TensorLifetime(birth=birth, death=death)

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        ins = self.producer_of(tensor)
        return self._proc.instrs.index(ins) if ins is not None else -1

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        "Get the step numbers of the steps that consume `tensor`."

        instrs = self._proc.instrs
        return [instrs.index(thunk) for thunk in self.consumers_of(tensor)]

    def producer_of(self, tensor: torch.Tensor, /) -> I | None:
        """
        Get the instr that produces the tensor.

        If it's from the input, return `None`.
        """

        if tensor in self._proc.inputs.tensors:
            return None

        return self[tensor].producer

    def consumers_of(self, tensor: torch.Tensor, /) -> cabc.Generator[I]:
        "Yield the instrs that consume the tensor."

        yield from self[tensor].consumers

    def _all_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from self.input_tensors
        yield from self.output_tensors

        yield from _all_thunk_tensors(self._proc.instrs)

    @property
    def input_tensors(self):
        return self.proc.inputs.tensors

    @property
    def output_tensors(self):
        return tree_leaves(self.proc.outputs)

    def _validate_input_output(self) -> None:
        proc = self._proc

        for input in proc.inputs.tensors:
            # Input should not have a producer.
            if not self[input].is_free:
                raise ValueError("Input is not a free variable.")

        for output in self.output_tensors:
            # Output is not produced.
            if output not in self:
                raise ValueError("Output not discovered.")

        if set(proc.inputs.tensors) & set(self.output_tensors):
            raise ValueError("Inputs are in the outputs. Not allowed yet.")


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
        self._consumers: AnySet[F] = any_set()

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
            self.consumers.add(consumer)

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
        yield from thunk.tensor_inputs
        yield from thunk.tensor_outputs


def _build_tensor_refs(
    thunks: cabc.Iterable[Instr], inputs: cabc.Sequence[torch.Tensor]
) -> dict[torch.Tensor, TensorRef]:
    "Get the mapping from id of `torch.Tensor` to corresponding tensor ref."

    # Register all inputs.
    mapping: dict[torch.Tensor, TensorRef] = _tensor_refs_inputs(inputs)

    # Register all outputs of thunks.
    mapping |= _tensor_refs_outputs(thunks)

    # Register each thunk's input.
    _link_inputs_for_mapping(mapping, thunks)

    return mapping


def _tensor_refs_inputs(
    inputs: cabc.Sequence[torch.Tensor],
) -> dict[torch.Tensor, TensorRef]:
    mapping: dict[torch.Tensor, TensorRef] = {}

    for input in inputs:
        ref = TensorRef(producer=None, tensor=input)
        mapping[ref.tensor] = ref

    return mapping


def _tensor_refs_outputs[I: Instr](
    thunks: cabc.Iterable[I],
) -> dict[torch.Tensor, TensorRef[I]]:
    mapping: dict[torch.Tensor, TensorRef[I]] = {}

    for thunk in thunks:
        for output in thunk.tensor_outputs:
            if output in mapping:
                raise ValueError("The output fake tensor is not unique.")

            ref = TensorRef(producer=thunk, tensor=output)
            mapping[ref.tensor] = ref
    return mapping


def _link_inputs_for_mapping(
    mapping: dict[torch.Tensor, TensorRef], thunks: cabc.Iterable[Instr]
) -> None:
    for thunk in thunks:
        assert isinstance(thunk, Instr)
        for input in thunk.tensor_inputs:
            assert input in mapping
            mapping[input].add_consumers(thunk)


def _used_outside[I: Instr](proc: Proc[I], selected: OrderedIndex) -> TList:
    """
    Add all tensors used outside of selected region.

    Args:
        proc: The procedure.
        selected: Indices that are selected.
    """

    used = TList()

    # Only check those that occur after, as this is a DAG.
    for i in range(selected.min() + 1, len(proc)):
        if i in selected:
            continue

        used |= proc[i].tensor_inputs

    # Populate the outputs as it's considered "consumed".
    used |= TList.deduplicate(proc.tindex.output_tensors)

    return used


# The querying related utilities ====


class _ProcTuple[I: Instr](typing.NamedTuple):
    "A namedtuple to avoid paying construction cost of `Proc`."

    ilist: InstrList[I]
    inputs: BoundArgs
    outputs: TList


def _replace_sub_proc[I: Instr](
    *, proc: Proc[I], query: Query[I], sub_proc: Proc[I]
) -> _ProcTuple[I]:
    """
    Replace the query with a new subset.
    """

    queried = _query_select(query=query, proc=proc)
    rplc, queried_idx = _tensor_rplc(proc=proc, queried=queried, sub_proc=sub_proc)

    # Drop the ones that are queried.
    new_instrs: list[I] = [
        thunk.tree_map_only(torch.Tensor, rplc)
        for i, thunk in enumerate(proc.instrs)
        if i not in queried_idx
    ]
    min_queried_idx = min(queried_idx)

    # Replace instructions.
    new_instrs = [
        *new_instrs[:min_queried_idx],
        *sub_proc.instrs,
        *new_instrs[min_queried_idx:],
    ]

    # Replace inputs and outputs.
    new_ins = tree_map_only(torch.Tensor, func=rplc, tree=proc.inputs)
    new_outs = tree_map_only(torch.Tensor, func=rplc, tree=proc.outputs)

    return _ProcTuple(
        ilist=InstrList.build(new_instrs),
        inputs=new_ins,
        outputs=new_outs,
    )


def _query_select[I: Instr](*, query: Query[I], proc: Proc[I]) -> _ProcTuple[I]:
    # Sorted and deduplicated, so `selected` is in step order.
    indices = query(proc)
    return _select_indices(selection=indices, prog=proc)


def _select_indices[I: Instr](
    *, selection: OrderedIndex, prog: Proc[I]
) -> _ProcTuple[I]:
    """
    Produce a subprogram whose:
    Input is any tensor used in this scope but not defined in the scope.
    Output is any tensor produced and used in downstream.
    """

    selected = prog.instrs[selection.tolist()]

    produced_here = functools.reduce(operator.or_, (i.tensor_outputs for i in selected))
    used_outside = _used_outside(prog, selection)

    # Inputs: used by our selected by not produced inside the region.
    inputs = TList.deduplicate(
        t for i in selected for t in i.tensor_inputs if t not in produced_here
    )

    # Outputs: produced by our selected and used by outside thunks.
    outputs = TList.deduplicate(
        t for i in selected for t in i.tensor_outputs if t in used_outside
    )

    # Inputs should not depend on intermediate.
    # Since `inputs_produced_by` represent intermediate, it can be empty.
    inputs_produced_by = [prog.tindex.output_of_step(t) for t in inputs]
    if inputs_produced_by and selection.min() < max(inputs_produced_by):
        raise ValueError("Illegal subset where input depend on intermediate.")

    return _ProcTuple(
        ilist=selected,
        inputs=BoundArgs.from_iters(inputs),
        outputs=outputs,
    )


def _tensor_rplc[I: Instr](
    proc: Proc[I], queried: _ProcTuple[I], sub_proc: Proc[I]
) -> tuple[cabc.Callable[[torch.Tensor], torch.Tensor], set[int]]:
    "Find the replacement tensors, before -> after."

    # Inputs are matched by position, parameter names are not compared.
    queried_ins = queried.inputs.tensors
    sub_ins = sub_proc.inputs.tensors

    if queried_ins.shapes() != sub_ins.shapes():
        raise ValueError("Inputs have different shapes.")

    if queried.outputs.shapes() != sub_proc.outputs.shapes():
        raise ValueError("Outputs have different shapes.")

    if queried_ins.dtypes() != sub_ins.dtypes():
        raise ValueError("Inputs have different dtypes.")

    if queried.outputs.dtypes() != sub_proc.outputs.dtypes():
        raise ValueError("Outputs have different dtypes.")

    # Get the indices of the queried subnet.
    queried_index = {proc.instrs.index(q) for q in queried.ilist}

    # Inputs and outputs are not shared.
    assert sub_ins.isdisjoint(sub_proc.outputs)

    # Build mapping for replacement.
    in_to_out = {}
    for before, after in zip(queried_ins, sub_ins):
        in_to_out[before] = after
    for before, after in zip(queried.outputs, sub_proc.outputs):
        in_to_out[before] = after

    def replacer(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(type(t))

        return in_to_out.get(t, t)

    return replacer, queried_index


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
