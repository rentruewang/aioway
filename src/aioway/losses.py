# Copyright (c) AIoWay Authors - All Rights Reserved

"Route the spaces to losses."

from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs


def route_loss(
    observ: tspecs.TensorSpec, action: tspecs.TensorSpec
) -> cabc.Generator[nn.Module]:
    """
    This is the API that mimicks `emit` for networks, but for losses.
    """

    if is_categorical(observ, action):
        yield nn.CrossEntropyLoss()

    if is_probability(observ, action):
        yield nn.KLDivLoss()

    if is_equal_bounded_cont(observ, action):
        yield nn.BCELoss()

    if is_bce_logits(observ, action):
        yield nn.BCEWithLogitsLoss()

    if is_equal_unbounded(observ, action):
        yield nn.MSELoss()
        yield nn.L1Loss()
        yield nn.SmoothL1Loss()
        yield nn.HuberLoss()


def is_categorical(observ, action):
    if not isinstance(observ, tspecs.BoundedContinuous):
        return False

    if not isinstance(action, tspecs.BoundedDiscrete):
        return False

    return observ.ndim == action.ndim + 1


def is_probability(observ, action):
    if not isinstance(observ, tspecs.OneHot):
        return False

    if not isinstance(action, tspecs.OneHot):
        return False

    return observ.shape == action.shape


def is_equal_bounded_cont(observ, action):
    if not isinstance(observ, tspecs.BoundedContinuous):
        return False

    if not isinstance(action, tspecs.BoundedContinuous):
        return False

    if observ.shape != action.shape:
        return False

    return observ.low == action.low == 0 and observ.high == action.high == 1


def is_bce_logits(observ, action):
    if not isinstance(observ, tspecs.Unbounded):
        return False

    if not isinstance(action, tspecs.BoundedContinuous):
        return False

    if observ.shape != action.shape:
        return False

    return action.low == 0 and action.high == 1


def is_equal_unbounded(observ, action):
    if not isinstance(observ, tspecs.Unbounded):
        return False

    if not isinstance(action, tspecs.Unbounded):
        return False

    return observ.shape == action.shape
