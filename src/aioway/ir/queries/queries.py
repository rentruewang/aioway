# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np

from aioway._utils import IntArray
from aioway.ir.progs import Program, Query

__all__ = ["IndexQuery"]


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
    def __call__(self, prog: Program, /) -> list[int]:
        idx: IntArray = np.asarray(self.indices)

        if (idx < 0).any():
            raise IndexError("Some indices are negative.")

        if (idx >= len(prog)).any():
            raise IndexError("Some indices are out of bounds.")

        return idx.tolist()
