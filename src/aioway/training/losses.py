# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import torch
from torch import nn

from aioway.schemas import Attr, Shape

__all__ = ["loss_func"]


class LossFunc(typing.Protocol):
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: ...


def loss_func(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Route the input and attr to their loss with simple heuristics.
    """

    input_attr, target_attr = map(Attr.parse, [input, target])
    input_shape, target_shape = input_attr.shape, target_attr.shape
    input_dtype, target_dtype = input_attr.dtype, target_attr.dtype
    input_family, target_family = input_dtype.family, target_dtype.family

    same_shape = input_shape.broadcastable(target_shape)

    if input_family != "float":
        raise ValueError("Cannot train a non float tensor.")

    if same_shape and target_family == "float":
        in_range = (target > 0) & (target < 1)

        if in_range.all():
            return nn.BCELoss()(input, target)

        else:
            return nn.MSELoss()(input, target)

    ce_shape = Shape.parse() if target_shape.ndim == 0 else target_shape.unsqueeze(1)
    if ce_shape.broadcastable(input_shape) and target_family == "int":
        return nn.CrossEntropyLoss()(input, target)

    raise NotImplementedError(f"{input=}, {target=}.")


def _bool_tensor_to_bce_target(target: torch.Tensor) -> torch.Tensor:
    target = target.float()
    target = (target - 1e-5).abs()
    assert (target > 0).all()
    assert (target < 1).all()
    return target
