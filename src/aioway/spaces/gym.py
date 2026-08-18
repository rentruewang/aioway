# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import numpy as np
import torch
from numpy import typing as npt
from torchrl.data import tensor_specs as tspecs

from aioway._torch import Device, DeviceLike, DType, DTypeLike, Shape, ShapeLike

from .spaces import TensorSpecSpace

__all__ = ["unbounded_box_space", "scalar_box_space", "array_box_space"]


def unbounded_box_space(
    shape: ShapeLike,
    *,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> TensorSpecSpace[tspecs.Unbounded]:
    """
    A continuous box spec.

    Args:
        shape: The shape of the tensors.
        dtype: The dtype of the values. Optional.
        device: The device of the tensors. Optional.

    Returns:
        An `Unbounded` spec.
        It would be continuous or discrete depending on the dtype.
    """

    return TensorSpecSpace(
        tspecs.Unbounded(
            shape=_parse_shape(shape),
            device=_exec_if_not_none(device, _parse_device),
            dtype=_exec_if_not_none(dtype, _parse_dtype),
        )
    )


def scalar_box_space(
    low: float,
    high: float,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> TensorSpecSpace[tspecs.Bounded]:
    """
    A continuous tensor bounded elementwise by `low` and `high`. Scalar version.

    Args:
        low: The scalar lower bound. Inclusive.
        high: The scalar higher bound. Inclusive.
        shape: The shape of the tensors.
        dtype: The dtype of the values.
        device: The device of the tensors.

    Returns:
        A `Bounded` spec.
    """

    return TensorSpecSpace(
        tspecs.Bounded(
            low=low,
            high=high,
            shape=_exec_if_not_none(shape, _parse_shape),
            device=_exec_if_not_none(device, _parse_device),
            dtype=_exec_if_not_none(dtype, _parse_dtype),
        )
    )


def array_box_space(
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> TensorSpecSpace[tspecs.Bounded]:
    """
    A continuous tensor bounded elementwise by `low` and `high`.
    Here each element of the shape can have different bounds.

    Args:
        low: The array lower bound. Inclusive.
        high: The array higher bound. Inclusive.
        shape: The shape of the tensors. Optional. Casts `low` and `high` shapes.
        dtype: The dtype of the values. Optional. Casts `low` and `high` dtypes.
        device: The device of the tensors. Optional. Casts `low` and `high` devices.

    Returns:
        A `Bounded` spec.
    """

    low_tensor = torch.from_numpy(np.asarray(low))
    high_tensor = torch.from_numpy(np.asarray(high))

    return TensorSpecSpace(
        tspecs.Bounded(
            low=low_tensor,
            high=high_tensor,
            device=_exec_if_not_none(device, _parse_device),
            shape=_exec_if_not_none(shape, _parse_shape),
            dtype=_exec_if_not_none(dtype, _parse_dtype),
        )
    )


@typing.overload
def _exec_if_not_none[I, O](item: None, func) -> None: ...


@typing.overload
def _exec_if_not_none[I, O](item: I, func: cabc.Callable[[I], O]) -> O: ...


def _exec_if_not_none[I, O](item: I | None, func: cabc.Callable[[I], O]) -> O | None:
    if item is None:
        return item

    return func(item)


def _parse_shape(shape: ShapeLike) -> torch.Size:
    return Shape.parse(shape).torch()


def _parse_device(device: DeviceLike) -> torch.device:
    return Device.parse(device).torch()


def _parse_dtype(dtype: DTypeLike) -> torch.dtype:
    return DType.parse(dtype).torch()
