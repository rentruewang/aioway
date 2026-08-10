# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import torch
from numpy import typing as npt
from torchrl import data as rldata

from aioway._torch import Device, DeviceLike, DType, DTypeLike, Shape, ShapeLike

__all__ = ["unbounded_box_spec", "scalar_box_spec", "array_box_spec"]


def unbounded_box_spec(
    *, shape: ShapeLike, dtype: DTypeLike, device: DeviceLike
) -> rldata.Unbounded:
    """
    A continuous box spec.

    Args:
        shape: The shape of the tensors.
        dtype: The dtype of the values.
        device: The device of the tensors.

    Returns:
        An `Unbounded` spec.
        It would be continuous or discrete depending on the dtype.
    """

    return rldata.Unbounded(
        shape=Shape.parse(shape).torch(),
        device=Device.parse(device).torch(),
        dtype=DType.parse(dtype).torch(),
    )


def scalar_box_spec(
    low: float, high: float, *, shape: ShapeLike, dtype: DTypeLike, device: DeviceLike
) -> rldata.Bounded:
    """
    A continuous tensor bounded elementwise by `low` and `high`. Scalar version.

    Args:
        low: The scalar lower bound.
        high: The scalar higher bound.
        shape: The shape of the tensors.
        dtype: The dtype of the values.
        device: The device of the tensors.

    Returns:
        A `Bounded` spec.
    """

    return rldata.Bounded(
        low=low,
        high=high,
        shape=Shape.parse(shape).torch(),
        device=Device.parse(device).torch(),
        dtype=DType.parse(dtype).torch(),
    )


def array_box_spec(
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    *,
    shape: ShapeLike,
    dtype: DTypeLike,
    device: DeviceLike,
) -> rldata.Bounded:
    """
    A continuous tensor bounded elementwise by `low` and `high`.
    Here each element of the shape can have different bounds.

    Args:
        low: The array lower bound.
        high: The array higher bound.
        shape: The shape of the tensors. Must match the `low` and `high` shapes.
        dtype: The dtype of the values. Must match the `low` and `high` shapes.
        device: The device of the tensors. Must match the `low` and `high` shapes.

    Returns:
        A `Bounded` spec.
    """

    torch_device = Device.parse(device).torch()
    torch_shape = Shape.parse(shape).torch()
    torch_dtype = DType.parse(dtype).torch()

    low_tensor = torch.from_numpy(np.asarray(low))
    high_tensor = torch.from_numpy(np.asarray(high))

    if low_tensor.shape != torch_shape:
        raise ValueError(f"{low_tensor.shape=} does not match {shape=}.")
    if high_tensor.shape != torch_shape:
        raise ValueError(f"{high_tensor.shape=} does not match {shape=}.")

    return rldata.Bounded(
        low=low_tensor,
        high=high_tensor,
        shape=torch_shape,
        device=torch_device,
        dtype=torch_dtype,
    )
