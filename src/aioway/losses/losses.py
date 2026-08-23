# Copyright (c) AIoWay Authors - All Rights Reserved

"Route the spaces to losses."

from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import TSpec

__all__ = ["route_loss"]


def route_loss(input: TSpec, target: TSpec) -> cabc.Generator[nn.Module]:
    """
    This is the API that mimicks `emit` for networks, but for losses.
    """

    if is_categorical(input, target):
        yield nn.CrossEntropyLoss()

    if is_probability(input, target):
        yield nn.KLDivLoss()

    if is_equal_bounded_cont(input, target):
        yield nn.BCELoss()

    if is_bce_logits(input, target):
        yield nn.BCEWithLogitsLoss()

    if is_equal_unbounded(input, target):
        yield nn.MSELoss()
        yield nn.L1Loss()
        yield nn.SmoothL1Loss()
        yield nn.HuberLoss()


def is_categorical(input, target):
    if not isinstance(input, tspecs.BoundedContinuous):
        return False

    if not isinstance(target, tspecs.BoundedDiscrete):
        return False

    return input.ndim == target.ndim + 1


def is_probability(input, target):
    if not isinstance(input, tspecs.OneHot):
        return False

    if not isinstance(target, tspecs.OneHot):
        return False

    return input.shape == target.shape


def is_equal_bounded_cont(input, target):
    if not isinstance(input, tspecs.BoundedContinuous):
        return False

    if not isinstance(target, tspecs.BoundedContinuous):
        return False

    if input.shape != target.shape:
        return False

    return input.low == target.low == 0 and input.high == target.high == 1


def is_bce_logits(input, target):
    if not isinstance(input, tspecs.Unbounded):
        return False

    if not isinstance(target, tspecs.BoundedContinuous):
        return False

    if input.shape != target.shape:
        return False

    return target.low == 0 and target.high == 1


def is_equal_unbounded(input, target):
    if not isinstance(input, tspecs.Unbounded):
        return False

    if not isinstance(target, tspecs.Unbounded):
        return False

    return input.shape == target.shape
