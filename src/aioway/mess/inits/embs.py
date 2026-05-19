# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

from torch import nn

from aioway._types import dcls_no_repr

from .inits import MessInit

__all__ = ["Embedding"]


@dcls_no_repr
class _BaseEmbedding(MessInit):
    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    num_embeddings: int
    "The size of the dictionary of embeddings."

    embedding_dim: int
    "The size of each embedding vector."

    @typing.override
    def _check_data(self):
        if self.num_embeddings <= 0:
            raise ValueError(f"{self.num_embeddings=} <= 0.")

        if self.embedding_dim <= 0:
            raise ValueError(f"{self.embedding_dim=} <= 0.")


@dcls_no_repr
class Embedding(_BaseEmbedding, key=nn.Embedding):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """


@dcls_no_repr
class EmbeddingBag(_BaseEmbedding, key=nn.EmbeddingBag):
    """
    Compute sums or means of 'bags' of embeddings,
    without instantiating the intermediate embeddings.
    """
