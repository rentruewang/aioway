# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from torch import nn

from aioway._common import dcls_frozen_no_repr

from .might import Might

__all__ = ["Embedding"]


@dcls_frozen_no_repr
class _BaseEmbedding(Might):
    KEY: typing.ClassVar[type[nn.Module]] = NotImplemented

    num_embeddings: int
    "The size of the dictionary of embeddings."

    embedding_dim: int
    "The size of each embedding vector."

    def __post_init__(self):
        if self.num_embeddings <= 0:
            raise ValueError(f"{self.num_embeddings=} <= 0.")

        if self.embedding_dim <= 0:
            raise ValueError(f"{self.embedding_dim=} <= 0.")


@dcls_frozen_no_repr
class Embedding(_BaseEmbedding):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

    KEY = nn.Embedding


@dcls_frozen_no_repr
class EmbeddingBag(_BaseEmbedding):
    """
    Compute sums or means of 'bags' of embeddings,
    without instantiating the intermediate embeddings.
    """

    KEY = nn.EmbeddingBag
