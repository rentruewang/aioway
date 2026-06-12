# Copyright (c) AIoWay Authors - All Rights Reserved

"The index (currently only `hnswlib`)."

import typing
from collections import abc as cabc

import numpy as np
import torch

from aioway._utils import FloatArray, UIntArray
from aioway.hop import TensorHop, hop_dcls

__all__ = ["HnswIndex", "HnswIndexHop"]


class HnswIndex:
    """
    An index exposing `hnswlib` API.
    This has a lot of defaults, change this in the future.
    """

    def __init__(
        self, dim: int, space: typing.Literal["cosine", "ip", "l2"] = "cosine"
    ):
        import hnswlib

        self.idx = hnswlib.Index(space=space, dim=dim)

    def train(self, x: FloatArray, ef_construction: int = 100, m: int = 10):
        self.idx.init_index(max_elements=len(x), ef_construction=ef_construction, M=m)
        self.idx.add_items(x, ids=np.arange(len(x)))

    def search(self, query: FloatArray, k: int = 1):
        if k <= 0:
            raise ValueError(f"The neighbor count {k} must be positive.")

        return self.idx.knn_query(query, k)

    @classmethod
    def from_tensors(cls, training: torch.Tensor) -> typing.Self:
        assert training.ndim == 2
        dim = training.shape[1]
        return cls(dim)

    @classmethod
    def from_hop(cls, tensors: TensorHop) -> typing.Self:
        training = torch.cat(list(tensors))
        return cls.from_tensors(training)


@hop_dcls
class HnswIndexHop(TensorHop):
    index: HnswIndex
    """
    The index backed by the `hnswlib` library.
    """

    source: torch.Tensor
    """
    The source that would be queried.
    """

    query: TensorHop
    """
    The querying tensor.
    """

    k: int
    """
    The number of neighbors to return.
    """

    def iterate(self) -> cabc.Generator[torch.Tensor]:
        for query in self.query:
            _, index = self.index.search(query)

            yield self.source[torch.from_numpy(index)]
