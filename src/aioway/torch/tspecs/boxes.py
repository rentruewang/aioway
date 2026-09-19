# Copyright (c) AIoWay Authors - All Rights Reserved


import numpy as np
import torch
from numpy import typing as npt
from torchrl.data import tensor_specs as tspecs

from aioway.torch import DeviceLike, DTypeLike, ShapeLike

from ._utils import exec_if_not_none, parse_device, parse_dtype, parse_shape

__all__ = ["unbounded_box_tspec", "scalar_box_tspec", "array_box_tspec"]


def unbounded_box_tspec(
    shape: ShapeLike,
    *,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> tspecs.Unbounded:
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

    return tspecs.Unbounded(
        shape=parse_shape(shape),
        device=exec_if_not_none(device, parse_device),
        dtype=exec_if_not_none(dtype, parse_dtype),
    )


def scalar_box_tspec(
    low: float,
    high: float,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> tspecs.Bounded:
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

    return tspecs.Bounded(
        low=low,
        high=high,
        shape=exec_if_not_none(shape, parse_shape),
        device=exec_if_not_none(device, parse_device),
        dtype=exec_if_not_none(dtype, parse_dtype),
    )


def array_box_tspec(
    low: npt.ArrayLike,
    high: npt.ArrayLike,
    *,
    shape: ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    device: DeviceLike | None = None,
) -> tspecs.Bounded:
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

    return tspecs.Bounded(
        low=low_tensor,
        high=high_tensor,
        device=exec_if_not_none(device, parse_device),
        shape=exec_if_not_none(shape, parse_shape),
        dtype=exec_if_not_none(dtype, parse_dtype),
    )
