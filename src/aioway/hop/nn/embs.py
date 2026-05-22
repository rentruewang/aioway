# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from .inits import NnInit, nn_init_dcls

__all__ = ["Embedding"]


@nn_init_dcls
class _BaseEmbedding(NnInit):
    NN: typing.ClassVar[type[nn.Module]] = NotImplemented

    num_embeddings: int
    "The size of the dictionary of embeddings."

    embedding_dim: int
    "The size of each embedding vector."

    def __post_init__(self):
        if self.num_embeddings <= 0:
            raise ValueError(f"{self.num_embeddings=} <= 0.")

        if self.embedding_dim <= 0:
            raise ValueError(f"{self.embedding_dim=} <= 0.")


@nn_init_dcls
class Embedding(_BaseEmbedding):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

    NN = nn.Embedding


@nn_init_dcls
class EmbeddingBag(_BaseEmbedding):
    """
    Compute sums or means of 'bags' of embeddings,
    without instantiating the intermediate embeddings.
    """

    NN = nn.EmbeddingBag
