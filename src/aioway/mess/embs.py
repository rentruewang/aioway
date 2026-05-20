# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from .fwds import InputFwd
from .mess import Mess, MessInit, mess_init_dcls

__all__ = []


@mess_init_dcls
class BaseEmbedding(MessInit):
    num_embeddings: int
    "The size of the dictionary of embeddings."

    embedding_dim: int
    "The size of each embedding vector."

    def __post_init__(self):
        if self.num_embeddings <= 0:
            raise ValueError(f"{self.num_embeddings=} <= 0.")

        if self.embedding_dim <= 0:
            raise ValueError(f"{self.embedding_dim=} <= 0.")


_ = Mess(nn_type=nn.Embedding, init=BaseEmbedding, fwd=InputFwd)
"""
A simple lookup table that stores embeddings of a fixed dictionary and size.
"""


_ = Mess(nn_type=nn.EmbeddingBag, init=BaseEmbedding, fwd=InputFwd)
"""
Compute sums or means of 'bags' of embeddings,
without instantiating the intermediate embeddings.
"""
