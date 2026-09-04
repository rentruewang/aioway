# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from .deductions import deduction_for

__all__ = ["dropout_deduct", "norm_deduct", "emb_deduct", "activation_deduct"]


@deduction_for(nn.ReLU).register
@deduction_for(nn.ReLU6).register
@deduction_for(nn.GELU).register
@deduction_for(nn.CELU).register
@deduction_for(nn.Tanh).register
@deduction_for(nn.Sigmoid).register
def activation_deduct(self, input: tspecs.Unbounded) -> tspecs.Unbounded:
    return input


@deduction_for(nn.Dropout1d).register
@deduction_for(nn.Dropout2d).register
@deduction_for(nn.Dropout3d).register
def dropout_deduct(self, input: tspecs.Unbounded) -> tspecs.Unbounded:
    return input


@deduction_for(nn.BatchNorm1d).register
@deduction_for(nn.BatchNorm2d).register
@deduction_for(nn.BatchNorm3d).register
@deduction_for(nn.InstanceNorm1d).register
@deduction_for(nn.InstanceNorm2d).register
@deduction_for(nn.InstanceNorm3d).register
@deduction_for(nn.LayerNorm).register
def norm_deduct(self, input: tspecs.Unbounded) -> tspecs.Unbounded:
    return input


@deduction_for(nn.Embedding).register
def emb_deduct(self: nn.Embedding, input: tspecs.Categorical) -> tspecs.Unbounded:
    shape = torch.Size([self.num_embeddings])
    return tspecs.Unbounded(shape=shape)
