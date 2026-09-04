# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from .deductions import deduction_for


@deduction_for(nn.Embedding).register
def emb_deduct(self: nn.Embedding, input: tspecs.Categorical) -> tspecs.Unbounded:
    shape = torch.Size([self.num_embeddings])
    return tspecs.Unbounded(shape=shape)
