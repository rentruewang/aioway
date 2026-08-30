# Copyright (c) AIoWay Authors - All Rights Reserved


from torch import nn

from ..infers import UnboundedInfer
from ..nn import NnInstr, instr_dcls

__all__ = ["Embedding"]


@instr_dcls
class _BaseEmbedding(NnInstr):
    num_embeddings: int
    "The size of the dictionary of embeddings."

    embedding_dim: int
    "The size of each embedding vector."

    def __post_init__(self):
        if self.num_embeddings <= 0:
            raise ValueError(f"{self.num_embeddings=} <= 0.")

        if self.embedding_dim <= 0:
            raise ValueError(f"{self.embedding_dim=} <= 0.")

    def __deduct__(self):
        return UnboundedInfer(self)


@instr_dcls
class Embedding(_BaseEmbedding):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

    NN = nn.Embedding


@instr_dcls
class EmbeddingBag(_BaseEmbedding):
    """
    Compute sums or means of 'bags' of embeddings,
    without instantiating the intermediate embeddings.
    """

    NN = nn.EmbeddingBag
