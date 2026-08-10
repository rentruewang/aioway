# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

import numpy as np
import torch
from numpy import typing as npt
from torchrl import data as rldata

from aioway._torch import Device, DeviceLike, DType, DTypeLike, Shape, ShapeLike

__all__ = ["unbounded_box_spec", "scalar_box_spec", "array_box_spec"]


def unbounded_box_spec(
    *,
    shape: ShapeLike,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> rldata.Unbounded:
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

    return rldata.Unbounded(
        shape=_parse_shape(shape),
        device=_exec_if_not_none(device, _parse_device),
        dtype=_exec_if_not_none(dtype, _parse_dtype),
    )


def scalar_box_spec(
    low: float,
    high: float,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> rldata.Bounded:
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

    return rldata.Bounded(
        low=low,
        high=high,
        shape=_exec_if_not_none(shape, _parse_shape),
        device=_exec_if_not_none(device, _parse_device),
        dtype=_exec_if_not_none(dtype, _parse_dtype),
    )


def array_box_spec(
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> rldata.Bounded:
    """
    A continuous tensor bounded elementwise by `low` and `high`.
    Here each element of the shape can have different bounds.

    Args:
        low: The array lower bound. Inclusive.
        high: The array higher bound. Inclusive.
        shape: The shape of the tensors. Optional. Must match the `low` and `high` shapes.
        dtype: The dtype of the values. Optional. Must match the `low` and `high` dtypes.
        device: The device of the tensors. Optional. Must match the `low` and `high` devices.

    Returns:
        A `Bounded` spec.
    """

    torch_device = _exec_if_not_none(device, _parse_device)
    torch_shape = _exec_if_not_none(shape, _parse_shape)
    torch_dtype = _exec_if_not_none(dtype, _parse_dtype)

    low_tensor = torch.from_numpy(np.asarray(low))
    high_tensor = torch.from_numpy(np.asarray(high))

    for attr_name, attr in [
        ("device", torch_device),
        ("shape", torch_shape),
        ("dtype", torch_dtype),
    ]:
        for tensor_name, tensor in [("low", low_tensor), ("high", high_tensor)]:
            # Since `attr` is optional, no checks if attr is `None`.
            if attr is None:
                continue

            if (tensor_attr := getattr(tensor, attr_name)) != attr:
                raise ValueError(
                    f"{tensor_name}_tensor.{attr_name}={tensor_attr} "
                    f"does not match excpected {attr=}."
                )

    return rldata.Bounded(
        low=low_tensor,
        high=high_tensor,
        shape=torch_shape,
        device=torch_device,
        dtype=torch_dtype,
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
