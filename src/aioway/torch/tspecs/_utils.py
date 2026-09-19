# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import torch

from aioway.torch.attrs import (
    Device,
    DeviceLike,
    DType,
    DTypeLike,
    Shape,
    ShapeLike,
)

__all__ = ["exec_if_not_none", "parse_device", "parse_dtype", "parse_shape"]


@typing.overload
def exec_if_not_none[I, O](item: None, func) -> None: ...


@typing.overload
def exec_if_not_none[I, O](item: I, func: cabc.Callable[[I], O]) -> O: ...


def exec_if_not_none[I, O](item: I | None, func: cabc.Callable[[I], O]) -> O | None:
    if item is None:
        return item

    return func(item)


def parse_shape(shape: ShapeLike) -> torch.Size:
    return Shape.parse(shape).torch()


def parse_device(device: DeviceLike) -> torch.device:
    return Device.parse(device).torch()


def parse_dtype(dtype: DTypeLike) -> torch.dtype:
    return DType.parse(dtype).torch()
