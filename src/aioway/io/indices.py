# Copyright (c) AIoWay Authors - All Rights Reserved

"The index (currently only faiss)."

import typing
from collections import abc as cabc

import torch, numpy as np

from aioway._utils import FloatArray, IntArray
from aioway.hop import TensorHop, hop_dcls

__all__ = ["FaissIndex", "FaissIndexHop"]


class FaissIndex:
    """
    An index exposing `hnswlib` API.
    """

    def __init__(self, dim: int, spec: str = "Flat"):
        import faiss

        self._idx = faiss.index_factory(dim, spec)
        self._dim = dim

    def train(self, x: FloatArray):
        self._idx.train(x)

    def search(self, query: FloatArray, k: int = 1) -> tuple[FloatArray, IntArray]:
        if not isinstance(query, np.ndarray):
            raise TypeError(f"{type(query)=} should be numpy.")

        if k <= 0:
            raise ValueError(f"The neighbor count {k} must be positive.")

        assert query.ndim == 2
        assert query.shape[-1] == self._dim

        breakpoint()
        return self._idx.search(query, k)

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
class FaissIndexHop(TensorHop):
    index: FaissIndex
    """
    The index backed by the `faiss` library.
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
            _, index = self.index.search(query.cpu().numpy())
            yield self.source[index]
