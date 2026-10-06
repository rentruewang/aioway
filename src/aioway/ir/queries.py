# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import AnySet, IntArray, any_set
from aioway.t import TList

from .instrs import Instr
from .progs import Program

__all__ = ["Query", "IndexQuery"]


class Query[I: Instr](abc.ABC):
    """
    A query is a subnet generator.
    """

    def __init__(self, base: type[I], /) -> None:
        self._base = base

    def select(self, prog: Program[I], /) -> Program[I]:
        """
        Produce a subset whose:
        Input is any tensor used in this scope but not defined in the scope.
        Output is any tensor produced and used in downstream.
        """

        # Sorted and deduplicated, so `selected` is in step order.
        indices = self._select_idx(prog)
        idx_set = set(indices.tolist())
        selected = prog[indices]

        produced_here = _get_tensor_sets(instr.outputs for instr in selected)
        used_outside = _used_outside(prog, idx_set)

        # Inputs: used by our selected by not produced inside the region.
        inputs = TList.from_iterable(
            t for instr in selected for t in instr.inputs if t not in produced_here
        )

        # Outputs: produced by our selected and used by outside thunks.
        outputs = TList.from_iterable(
            t for instr in selected for t in instr.outputs if t in used_outside
        )

        # Inputs should not depend on intermediate.
        # Since `inputs_produced_by` represent intermediate, it can be empty.
        inputs_produced_by = [prog.output_of_step(t) for t in inputs]
        if inputs_produced_by and indices.min() < max(inputs_produced_by):
            raise ValueError("Illegal subset where input depend on intermediate.")

        return Program(selected, inputs, outputs)

    @abc.abstractmethod
    def _select_idx(self, iset: Program, /) -> IntArray:
        raise NotImplementedError

    def rewrite(self, prog: Program[I], subset: Program[I]) -> Program[I]:
        prog = prog.copy()
        prog[self] = subset
        return prog

    @property
    def base(self) -> type[I]:
        return self._base


# Some implementations ====


@dcls.dataclass(frozen=True)
class IndexQuery(Query):
    """
    Query with subset of index.

    Raises:
        IndexError: if the graph index is out of bounds.
        ValueError: if the subgraph depends on intermediate value.
    """

    indices: list[int] | IntArray
    """
    The index to preserve. Indices must be within `[0, len)` for each instruction set.
    """

    @typing.override
    def _select_idx(self, iset: Program, /) -> IntArray:
        idx: IntArray = np.asarray(self.indices)

        if (idx < 0).any():
            raise IndexError("Some indices are negative.")

        if (idx >= len(iset)).any():
            raise IndexError("Some indices are out of bounds.")

        return idx


# Helper functions ====


def _used_outside(prog: Program, selected_idx: set[int]) -> AnySet[torch.Tensor]:
    """
    Add all tensors used outside of selected region.
    """

    used: AnySet[torch.Tensor] = any_set(torch.Tensor)

    # Only check those that occur after, as this is a DAG.
    for i in range(min(selected_idx), len(prog)):
        instr = prog[i]

        if i in selected_idx:
            continue

        for tensor in instr.inputs:
            used.add(tensor)

    for tensor in prog.outputs:
        used.add(tensor)

    return used


def _get_tensor_sets(tlists: cabc.Iterable[TList]) -> AnySet[torch.Tensor]:
    aset = any_set(torch.Tensor)

    for tlist in tlists:
        for tensor in tlist:
            aset.add(tensor)

    return aset
