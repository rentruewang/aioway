# Copyright (c) AIoWay Authors - All Rights Reserved

"Route the spaces to losses."
from aioway.emits import emitter_function

from collections import abc as cabc

from torch import nn
from torchrl.data import tensor_specs as tspecs

from aioway.tspecs import (
    TSpecLike,
    as_tspec,
    ArgsTSpec,
    TSpec,
    is_tspec_like,
    LossTSpec,
)

__all__ = ["emit_loss", "route_loss"]

_INPUT = "input"
_TARGET = "target"


@emitter_function
def emit_loss(observ: TSpec, action: TSpec) -> nn.Module:
    "Get the next loss type with an `Emitter` API."

    if not isinstance(observ, ArgsTSpec):
        return NotImplemented

    if not isinstance(action, LossTSpec):
        return NotImplemented

    input_spec = observ.get(_INPUT)
    target_spec = observ.get(_TARGET)

    if not is_tspec_like(input_spec) or not is_tspec_like(target_spec):
        return NotImplemented

    try:
        return next(route_loss(input_spec, target_spec))
    except StopIteration:
        return NotImplemented


def route_loss(input: TSpecLike, target: TSpecLike) -> cabc.Generator[nn.Module]:
    """
    This is the API that mimicks `emit` for networks, but for losses.
    """

    input = as_tspec(input)
    target = as_tspec(target)

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
