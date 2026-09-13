# Copyright (c) AIoWay Authors - All Rights Reserved

"The modules in the `torch.nn` package."

import dataclasses as dcls
import logging
import typing

import torch
from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.nn.deductions import deduction_for
from aioway.nn.tspecs import LossTSpec, TSpec, sample_from_tspec
from aioway.tensors import fake_mode

__all__ = [
    "sequential_deduct",
    "identity_deduct",
    "flatten_deduct",
    "unflatten_deduct",
    "linear_deduct",
    "bilinear_deduct",
    "dropout_deduct",
    "norm_deduct",
    "emb_deduct",
    "activation_deduct",
    "SlidingDeduction",
    "symmetric_loss_deduct",
]

LOGGER = logging.getLogger(__name__)

_LOSS_TSPEC = LossTSpec()


@deduction_for(nn.Sequential).register
def sequential_deduct(self: nn.Sequential, input: TSpec) -> TSpec:
    for sub in self.children():
        deduction = deduction_for(sub)

        if (output := deduction(sub, input)) is NotImplemented:
            return NotImplemented

        input = output
    return input


@deduction_for(nn.Identity).register
def identity_deduct(self, input: TSpec) -> TSpec:
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


@dcls.dataclass(frozen=True)
class SlidingDeduction:
    allowed_dims: tuple[int, ...]
    "The allowed dimensions."

    # Mark `module` as `typing.Any` to use with multiple types.
    def __call__(self, module: typing.Any, input: tspecs.Unbounded) -> tspecs.Unbounded:
        if input.ndim not in self.allowed_dims:
            return NotImplemented

        with fake_mode():
            module = module.module()
            output: torch.Tensor = module(sample_from_tspec(input))

        batch, *rest = output.shape
        assert batch == input.shape[0]

        return tspecs.Unbounded(torch.Size(rest))

    def register(self, *modules: type[nn.Module]) -> None:
        for module in modules:
            deduction_for(module).register(self)


SlidingDeduction((1, 2)).register(nn.Conv1d, nn.AvgPool1d, nn.MaxPool1d)
SlidingDeduction((3,)).register(nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d)
SlidingDeduction((4,)).register(nn.Conv3d, nn.AvgPool3d, nn.MaxPool3d)


@deduction_for(nn.L1Loss).register
@deduction_for(nn.SmoothL1Loss).register
@deduction_for(nn.MSELoss).register
def symmetric_loss_deduct(
    self, input: tspecs.Unbounded, target: tspecs.Unbounded
) -> LossTSpec:
    if input.shape != target.shape:
        return NotImplemented

    return _LOSS_TSPEC
