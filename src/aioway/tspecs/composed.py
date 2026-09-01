# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import tensordict as td
from torchrl.data import tensor_specs as tspecs

__all__ = ["tcls_to_unbounded_tspec", "is_tcls_type"]


def tcls_to_unbounded_tspec(tcls: type[td.TensorClass], /) -> tspecs.Composite:
    """
    Convert from a `type[td.TensorClass]` to a `tspecs.Composite`,
    where it can directly be used to check.
    """

    if not is_tcls_type(tcls):
        raise TypeError(f"{tcls=} is not a `td.tensorclass`.")

    raise NotImplementedError


def is_tcls_type(tcls, /) -> typing.TypeIs[type[td.TensorClass]]:
    """
    Check if the argument is a `type`, and a `TensorClass`.
    """

    if not isinstance(tcls, type):
        return False

    if not dcls.is_dataclass(tcls):
        return False

    if not td.is_tensor_collection(tcls):
        return False

    return True
