# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls

import numpy as np

from aioway._utils import IntArray

from .sets import InstrSet

__all__ = ["Query", "IndexQuery"]


class Query(abc.ABC):
    """
    A query is a subnet generator.
    """

    def __call__(self, dag: InstrSet) -> InstrSet:
        raise NotImplementedError


# Some implementations ====


@dcls.dataclass(frozen=True)
class IndexQuery(Query):
    indices: list[int] | IntArray
    """
    The index to preserve. Indices must be within `[0, len)` for each instruction set.
    """

    def __call__(self, iset: InstrSet) -> InstrSet:
        idx: IntArray = np.asarray(self.indices)

        if (idx < 0).any():
            raise IndexError("Some indices are negative.")

        if (idx >= len(iset)).any():
            raise IndexError("Some indices are out of bounds.")

        result = InstrSet.from_thunk_list(iset[idx])

        # Check if input is not produced by intermediate steps,
        # which may be output of the subnet itself.
        inputs_produced_by = [iset.output_of_step(t) for t in result.inputs]
        if idx.min() < max(inputs_produced_by):
            raise IndexError("Illegal subset where input depend on intermediate.")

        return result
