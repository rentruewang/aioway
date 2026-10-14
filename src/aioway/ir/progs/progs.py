# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Program` that supports analysis."

import copy
import dataclasses as dcls
import functools
import operator
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._pytrees import KeyEntry, MappingKey, tree_map_only
from aioway._utils import AnyDict, AnySet, IntArray, any_dict, any_set, is_list_of
from aioway.ir.instrs import Instr, InstrList
from aioway.ir.queries import OrderedIndex, Query
from aioway.t import (
    BoundArgs,
    PyTreeFlatten,
    PyTreeFlattenWithKey,
    PyTreeMixin,
    Sign,
    TList,
    all_real,
    parse_attr,
)

__all__ = ["TensorRef", "Program", "ProgramIn", "TensorIndex", "TensorLifetime"]

# The Program class ====


class Program[I: Instr = typing.Any]:
    """
    A program is a DAG of callables, that are linked by fake tensors.

    The program owns the instructions and its signature (inputs and outputs).
    Tensor-level bookkeeping (producers, consumers, lifetimes) lives in
    `TensorIndex`, available as `Program.tindex`.
    """

    INSTR: typing.ClassVar[type[Instr]] = Instr
    "The instr base class. If not specified, anything is ok."

    def __init__(
        self,
        *,
        instrs: cabc.Iterable[I],
        inputs: ProgramIn,
        outputs: TList,
    ) -> None:
        if not isinstance(inputs, ProgramIn):
            raise TypeError(f"Inputs should be a `ProgramIn`. Got {type(inputs)=}.")

        if not isinstance(outputs, TList):
            raise TypeError(f"Outputs should be a `TList`. Got {type(outputs)=}.")

        self._instrs = InstrList.build(instrs)

        self._inputs = inputs
        self._outputs = outputs

        # Link producers / consumers, and validate inputs and outputs.
        self.reset_tindex()

        if (base := self.instrs.base) and not issubclass(base, self.INSTR):
            raise TypeError(
                f"The base of instr: {self.instrs.base} is not {self.INSTR}"
            )

    def __repr__(self) -> str:
        return repr(self.instrs)

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

    def __setitem__(self, query: Query, subset: Program) -> None:
        mutated = _replace_sub_prog(prog=self, query=query, sub_prog=subset)

        # Overwrite the references s.t. underlying data is not touched.
        # This is not supposed to fail. Also rebuilds the tensor index.
        self.__init__(
            instrs=mutated.ilist, inputs=mutated.inputs, outputs=mutated.outputs
        )

    @property
    def instrs(self) -> InstrList[I]:
        return self._instrs

    @property
    def inputs(self) -> ProgramIn:
        "The inputs, bound to the program's signature. Flat tensors: `.tensors`."
        return self._inputs

    @property
    def outputs(self) -> TList:
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

        consumed = TList.empty()
        produced = TList.empty()

        for instr in instrs:
            consumed += instr.tensor_inputs

            assert produced.isdisjoint(outs := instr.tensor_outputs)
            produced += outs

        return cls(
            instrs=instrs,
            inputs=ProgramIn.from_iters(consumed - produced),
            outputs=produced - consumed,
        )


# The ProgramIn class ====


class ProgramIn(PyTreeMixin):
    """
    The inputs of a `Program`: a `BoundArgs` whose arguments are all tensors.

    It is ordered and a mapping: every parameter is bound to exactly one fake
    tensor, looked up by name (`pin["x"]`) or by position (`pin[0]`).
    The flat tensors, in signature order, are in `.tensors`.

    The `pyt` children are the tensors in signature order, keyed by parameter name.
    The context is the `Sign`.
    """

    def __init__(self, bound: BoundArgs) -> None:
        self._bound = bound

        for param in self.sign.param_list:
            if param.is_var_positional or param.is_var_keyword:
                raise TypeError(
                    f"Variadic parameters are not (yet) supported: {param}."
                )

    def __contains__(self, obj: torch.Tensor) -> bool:
        return obj in self.bound.tensors

    def __repr__(self) -> str:
        return repr(self._bound)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ProgramIn):
            return NotImplemented

        # Tensors are compared by identity, as `==` on tensors is elementwise.
        return (
            True
            and self.sign == other.sign
            and all(a is b for a, b in zip(self, other))
        )

    def __iter__(self):
        return iter(self.values())

    def __hash__(self) -> int:
        return hash(self._bound)

    def __len__(self) -> int:
        return len(self.bound)

    def __getitem__(self, key: int | str) -> typing.Any:
        "The tensor bound to a parameter, by name or by position."

        return self.bound[key]

    def keys(self) -> cabc.KeysView[str]:
        "The keys. Only the arguments that are provided."

        return self.bound.keys()

    def values(self) -> cabc.ValuesView[typing.Any]:
        "The values. Only the arguments that are provided."

        return self.bound.values()

    def items(self) -> cabc.ItemsView[str, typing.Any]:
        "The items. Only the arguments that are provided."

        return self.bound.items()

    def rebind(self, *args, **kwargs) -> BoundArgs:
        "Rebind another set of inputs."
        return self.sign.bind(*args, **kwargs)

    @property
    def bound(self) -> BoundArgs:
        return self._bound

    @property
    def arguments(self):
        return self.bound.arguments

    @functools.cached_property
    def names(self) -> tuple[str, ...]:
        "The parameter names, in signature order."
        return tuple(param.name for param in self.bound.param_list)

    @property
    def sign(self) -> Sign:
        return self.bound.sign

    @property
    def arguments(self) -> dict[str, torch.Tensor]:
        "A copy of the tensors by parameter name, in signature order."
        return self.bound.arguments

    @property
    def tensors(self) -> TList:
        return self.bound.tensors

    @property
    def signature(self) -> Sign:
        return self.sign

    @property
    def args(self) -> tuple[torch.Tensor, ...]:
        return self.bound.args

    @property
    def kwargs(self) -> dict[str, torch.Tensor]:
        return self.bound.kwargs

    def _tree_flatten(self) -> PyTreeFlatten:
        return list(self.values()), self.bound.sign

    def _tree_flatten_with_keys(self) -> PyTreeFlattenWithKey:
        with_path = [(_map_key(name), self.arguments[name]) for name in self.names]
        return with_path, self.bound.sign

    @classmethod
    def _tree_unflatten(cls, children: cabc.Iterable, context) -> typing.Self:
        # Every parameter is bound, so the signature also gives the names.
        assert isinstance(context, Sign)

        arguments = dict(zip(context.params, children, strict=True))
        return cls(bound=BoundArgs.from_no_bind(sign=context, arguments=arguments))

    @classmethod
    def from_sign(
        cls, sign: Sign, /, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> typing.Self:
        """
        Bind tensors to `sign`, like `Sign.bind`.

        Raises:
            TypeError: if binding failed.
        """

        return cls(bound=sign.bind(*args, **kwargs))

    @classmethod
    def from_dict(cls, mapping: cabc.Mapping[str, typing.Any], /) -> typing.Self:
        "Bind tensors by name, in the order of the mapping."

        return cls.from_sign(Sign.from_inputs(mapping.keys()), **mapping)

    @classmethod
    def from_iters(
        cls, items: cabc.Iterable[typing.Any], /, *, prefix: str = "t"
    ) -> typing.Self:
        "Bind tensors positionally, to parameters named `t0`, `t1`, ..."

        tensors = tuple(items)
        names = [f"{prefix}{i}" for i in range(len(tensors))]
        return cls.from_sign(Sign.from_inputs(names), *tensors)


def _map_key(name: str) -> KeyEntry:
    return typing.cast(typing.Any, MappingKey)(name)


# The TensorIndex class ====


class TensorIndex[I: Instr = typing.Any]:
    """
    Tensor-level index over a `Program`,
    to decouple input output from lifetimes.

    Maps every tensor to a `TensorRef` (its producer and consumers),
    and provide info step and lifetime queries about tensors.

    `Program` builds this index on construction.
    """

    def __init__(self, program: Program[I], /) -> None:
        self._program = program

        self._tensors = TList.from_iterable(self._all_tensors())
        self._links = _build_tensor_refs(program.instrs, list(program.inputs.tensors))

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
    def program(self) -> Program[I]:
        "The program being indexed."
        return self._program

    @property
    def tensors(self) -> TList:
        "The tensors that exist in the instructions, inputs and outputs."
        return self._tensors

    @property
    def links(self) -> AnyDict[torch.Tensor, TensorRef[I]]:
        "Mapping from tensors to refs (linking functions)."
        return self._links

    def life(self, t: torch.Tensor, /) -> TensorLifetime:
        """
        Compute the lifetime of a tensor.

        birth = at which point we start keeping track of it.
        death = at which point it is no longer needed.
        """

        prog = self._program
        birth = self.output_of_step(t) if t not in prog.inputs.tensors else -1
        death = max(self.input_to_step(t)) if t not in prog.outputs else len(prog)
        return TensorLifetime(birth=birth, death=death)

    def output_of_step(self, tensor: torch.Tensor) -> int:
        """
        Get the step number of step that produced output.

        If not set (producer is None), return -1.
        """

        ins = self.producer_of(tensor)
        return self._program.instrs.index(ins) if ins is not None else -1

    def input_to_step(self, tensor: torch.Tensor) -> cabc.Sequence[int]:
        "Get the step numbers of the steps that consume `tensor`."

        instrs = self._program.instrs
        return [instrs.index(thunk) for thunk in self.consumers_of(tensor)]

    def producer_of(self, tensor: torch.Tensor, /) -> I | None:
        """
        Get the instr that produces the tensor.

        If it's from the input, return `None`.
        """

        if tensor in self._program.inputs.tensors:
            return None

        return self[tensor].producer

    def consumers_of(self, tensor: torch.Tensor, /) -> cabc.Generator[I]:
        "Yield the instrs that consume the tensor."

        yield from self[tensor].consumers

    def _all_tensors(self) -> cabc.Generator[torch.Tensor]:
        yield from self._program.inputs.tensors
        yield from self._program.outputs

        yield from _all_thunk_tensors(self._program.instrs)

    def _validate_input_output(self) -> None:
        prog = self._program

        for input in prog.inputs.tensors:
            # Input should not have a producer.
            if not self[input].is_free:
                raise ValueError("Input is not a free variable.")

        for output in prog.outputs:
            # Output is not produced.
            if output not in self:
                raise ValueError("Output not discovered.")

        if set(prog.inputs.tensors) & set(prog.outputs):
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
        yield from thunk.tensor_inputs
        yield from thunk.tensor_outputs


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
        for output in thunk.tensor_outputs:
            if output in mapping:
                raise ValueError("The output fake tensor is not unique.")

            ref = TensorRef(producer=thunk, tensor=output)
            mapping[ref.tensor] = ref
    return mapping


def _link_inputs_for_mapping(
    mapping: AnyDict[torch.Tensor, TensorRef], thunks: cabc.Iterable[Instr]
) -> None:
    for thunk in thunks:
        for input in thunk.tensor_inputs:
            assert input in mapping
            mapping[input].add_consumers(thunk)


def _used_outside[I: Instr](prog: Program[I], selected: OrderedIndex) -> TList:
    """
    Add all tensors used outside of selected region.

    Args:
        prog: The program.
        selected: Indices that are selected.
    """

    used = TList.empty()

    # Only check those that occur after, as this is a DAG.
    for i in range(selected.min() + 1, len(prog)):
        if i in selected:
            continue

        used += prog[i].tensor_inputs

    # Populate the outputs as it's considered "consumed".
    used += prog.outputs

    return used


# The querying related utilities ====


class _ProgramTuple[I: Instr](typing.NamedTuple):
    "A namedtuple to avoid paying construction cost of `Program`."

    ilist: InstrList[I]
    inputs: ProgramIn
    outputs: TList


def _replace_sub_prog[I: Instr](
    *, prog: Program[I], query: Query[I], sub_prog: Program[I]
) -> _ProgramTuple[I]:
    """
    Replace the query with a new subset.
    """

    queried = _query_select(query=query, prog=prog)
    rplc, queried_idx = _tensor_rplc(prog=prog, queried=queried, sub_prog=sub_prog)

    # Drop the ones that are queried.
    new_instrs: list[I] = [
        thunk.tree_map_only(torch.Tensor, rplc)
        for i, thunk in enumerate(prog.instrs)
        if i not in queried_idx
    ]
    min_queried_idx = min(queried_idx)

    # Replace instructions.
    new_instrs = [
        *new_instrs[:min_queried_idx],
        *sub_prog.instrs,
        *new_instrs[min_queried_idx:],
    ]

    # Replace inputs and outputs.
    new_ins = tree_map_only(torch.Tensor, func=rplc, tree=prog.inputs)
    new_outs = tree_map_only(torch.Tensor, func=rplc, tree=prog.outputs)

    return _ProgramTuple(
        ilist=InstrList.build(new_instrs),
        inputs=new_ins,
        outputs=new_outs,
    )


def _query_select[I: Instr](*, query: Query[I], prog: Program[I]) -> _ProgramTuple[I]:
    # Sorted and deduplicated, so `selected` is in step order.
    indices = query(prog)
    return _select_indices(selection=indices, prog=prog)


def _select_indices[I: Instr](
    *, selection: OrderedIndex, prog: Program[I]
) -> _ProgramTuple[I]:
    """
    Produce a subprogram whose:
    Input is any tensor used in this scope but not defined in the scope.
    Output is any tensor produced and used in downstream.
    """

    selected = prog.instrs[selection.tolist()]

    # Using `TList` as it has fast lookups and ergonomic operations.
    produced_here = functools.reduce(operator.add, (i.tensor_outputs for i in selected))
    used_outside = _used_outside(prog, selection)

    # Inputs: used by our selected by not produced inside the region.
    inputs = TList.from_iterable(
        t for i in selected for t in i.tensor_inputs if t not in produced_here
    )

    # Outputs: produced by our selected and used by outside thunks.
    outputs = TList.from_iterable(
        t for i in selected for t in i.tensor_outputs if t in used_outside
    )

    # Inputs should not depend on intermediate.
    # Since `inputs_produced_by` represent intermediate, it can be empty.
    inputs_produced_by = [prog.tindex.output_of_step(t) for t in inputs]
    if inputs_produced_by and selection.min() < max(inputs_produced_by):
        raise ValueError("Illegal subset where input depend on intermediate.")

    return _ProgramTuple(
        ilist=selected,
        inputs=ProgramIn.from_iters(inputs),
        outputs=outputs,
    )


def _tensor_rplc[I: Instr](
    prog: Program[I], queried: _ProgramTuple[I], sub_prog: Program[I]
) -> tuple[cabc.Callable[[torch.Tensor], torch.Tensor], set[int]]:
    "Find the replacement tensors, before -> after."

    # Inputs are matched by position, parameter names are not compared.
    queried_ins = queried.inputs.tensors
    sub_ins = sub_prog.inputs.tensors

    if queried_ins.shapes() != sub_ins.shapes():
        raise ValueError("Inputs have different shapes.")

    if queried.outputs.shapes() != sub_prog.outputs.shapes():
        raise ValueError("Outputs have different shapes.")

    if queried_ins.dtypes() != sub_ins.dtypes():
        raise ValueError("Inputs have different dtypes.")

    if queried.outputs.dtypes() != sub_prog.outputs.dtypes():
        raise ValueError("Outputs have different dtypes.")

    # Get the indices of the queried subnet.
    queried_index = {prog.instrs.index(q) for q in queried.ilist}

    # Inputs and outputs are not shared.
    assert sub_ins.keys().isdisjoint(sub_prog.outputs.keys())

    # Build mapping for replacement.
    in_to_out = any_dict(torch.Tensor)
    for before, after in zip(queried_ins, sub_ins):
        in_to_out[before] = after
    for before, after in zip(queried.outputs, sub_prog.outputs):
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
