# Copyright (c) AIoWay Authors - All Rights Reserved


import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.deductions import deduction_for

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


@instr_dcls
class Embedding(_BaseEmbedding):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """

    NN = nn.Embedding


@deduction_for(nn.Embedding).register
def emb_deduct(self: nn.Embedding, input: tspecs.Categorical) -> tspecs.Unbounded:
    shape = torch.Size([self.num_embeddings])
    return tspecs.Unbounded(shape=shape)


@instr_dcls
class EmbeddingBag(_BaseEmbedding):
    """
    Compute sums or means of 'bags' of embeddings,
    without instantiating the intermediate embeddings.
    """

    NN = nn.EmbeddingBag
