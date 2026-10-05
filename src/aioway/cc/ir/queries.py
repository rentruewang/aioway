# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing

import numpy as np
import torch

from aioway._utils import IntArray, any_dict

from .instrs import FCall
from .sets import InstrSet

__all__ = ["Query", "IndexQuery"]


class Query(abc.ABC):
    """
    A query is a subnet generator.
    """

    def select(self, iset: InstrSet, /) -> InstrSet:
        raise NotImplementedError

    def rewrite(self, iset: InstrSet, subset: InstrSet) -> InstrSet:
        return _replace_subset(query=self, iset=iset, subset=subset)


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
    def select(self, iset: InstrSet, /) -> InstrSet:
        idx: IntArray = np.asarray(self.indices)

        if (idx < 0).any():
            raise IndexError("Some indices are negative.")

        if (idx >= len(iset)).any():
            raise IndexError("Some indices are out of bounds.")

        result = InstrSet.from_thunk_list(iset[idx])

        # Inputs should not depend on intermediate.
        # Since `inputs_produced_by` represent intermediate, it can be empty.
        inputs_produced_by = [iset.output_of_step(t) for t in result.inputs]
        if inputs_produced_by and idx.min() < max(inputs_produced_by):
            raise ValueError("Illegal subset where input depend on intermediate.")

        # Output should not produce intermediate not used (but may be empty).
        # Since `outputs_consumed_by` represent intermediate, it can be empty.
        outputs_consumed_by = [s for t in result.outputs for s in iset.input_to_step(t)]
        if outputs_consumed_by and idx.max() > max(outputs_consumed_by):
            raise ValueError("Illegal subset where outputs intermediate.")

        return result


def _replace_subset(*, iset: InstrSet, query: Query, subset: InstrSet) -> InstrSet:
    """
    Replace the query with a new subset.
    """

    queried = query.select(iset)

    if queried.inputs.attrs() != subset.inputs.attrs():
        raise ValueError("Inputs are not compatible.")

    if queried.outputs.attrs() != subset.outputs.attrs():
        raise ValueError("Outputs are not compatible.")

    # Get the indices of the queried subnet and minimum (useful in inserting).
    qidx = {iset.index(q) for q in queried}
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
    new_instrs: list[FCall] = [
        thunk.tree_map_only(torch.Tensor, lambda t: in_to_out.get(t, t))
        for i, thunk in enumerate(iset.instrs)
        if i not in qidx
    ]

    # Replace with new.
    pre = new_instrs[:min_qidx]
    post = new_instrs[min_qidx:]
    return iset.from_thunk_list([*pre, *subset.instrs, *post])
