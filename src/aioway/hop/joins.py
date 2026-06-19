# Copyright (c) AIoWay Authors - All Rights Reserved

"The binary `Iter`s that consumes 2 `Iter`s."

import typing
from collections import abc as cabc

import tensordict as td
import torch

from .hop import BoundedIter, TdictIter, iter_dcls

__all__ = ["ZipIter", "NestedLoopJoinIter"]


@iter_dcls
class ZipIter(TdictIter):
    """
    `ZipStream` is similar to what `zip` does.
    """

    left: TdictIter
    """
    The LHS stream.
    """

    right: TdictIter
    """
    The RHS stream.
    """

    @property
    @typing.override
    def size(self) -> int:
        return min(self.left.size, self.right.size)

    @typing.override
    def iterate(self):
        for left_batch, right_batch in zip(self.left, self.right):
            yield td.merge_tensordicts(left_batch, right_batch)


@iter_dcls
class NestedLoopJoinIter(TdictIter):
    """
    This is a stream that combines 2 input streams in a nested-loop matter,
    as in `[[x, y] for x in left for y in right if x.key == y.key]`.

    The end result would be merged with `tensordict.merge_tensordicts`.
    """

    left: TdictIter
    """
    LHS is a normal stream. Will only be iterated over once.
    """

    right: BoundedIter
    """
    RHS is a `Stream` supporting index access, thus requiring materialization.
    """

    key: str
    """
    The key to join on.
    """

    @property
    @typing.override
    def size(self) -> int:
        return self.left.size * self.right.size

    @typing.override
    def iterate(self) -> cabc.Generator[td.TensorDict]:
        for lhs_batch in self.left:
            for rhs_batch in self.right:
                yield self.__iter_batch(lhs_batch, rhs_batch)

    def __iter_batch(self, lhs_batch: td.TensorDict, rhs_batch: td.TensorDict):
        lhs_select = lhs_batch[self.key]
        rhs_select = rhs_batch[self.key]

        assert isinstance(lhs_select, torch.Tensor), type(lhs_select)
        assert isinstance(rhs_select, torch.Tensor), type(rhs_select)

        matrix = lhs_select.data[:, None] == rhs_select.data[None, :]
        l, r = torch.nonzero(matrix).T
        assert len(l) == len(r) == torch.sum(matrix)
        out = td.merge_tensordicts(lhs_batch[l], rhs_batch[r])
        assert len(out) == torch.sum(matrix)
        return out
