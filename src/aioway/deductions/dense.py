# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from .deductions import deduction_for


@deduction_for(nn.Identity).register
def identity_deduct(self, input):
    return input


@deduction_for(nn.Flatten).register
def flatten_deduct(self, input: tspecs.Unbounded):
    return input.flatten(0, -1)


@deduction_for(nn.Unflatten).register
@typing.no_type_check
def unflatten_deduct(self: nn.Unflatten, input: tspecs.Unbounded):
    return input.unflatten(self.dim, sizes=torch.Size(self.unflattened_size))


@deduction_for(nn.Linear).register
def linear_deduct(linear: nn.Linear, input: tspecs.Unbounded) -> tspecs.Unbounded:
    assert input.shape[-1] == linear.in_features
    return tspecs.Unbounded(
        shape=torch.Size([*input.shape[:-1], linear.out_features]), dtype=input.dtype
    )


@deduction_for(nn.Bilinear).register
def bilinear_deduct(
    self: nn.Bilinear, input1: tspecs.Unbounded, input2: tspecs.Unbounded
) -> tspecs.Unbounded:
    input1_shape = input1.shape
    input2_shape = input2.shape

    if input1_shape[:-1] != input2_shape[:-1]:
        return NotImplemented

    return tspecs.Unbounded(shape=torch.Size([*input1_shape[:-1], self.out_features]))
