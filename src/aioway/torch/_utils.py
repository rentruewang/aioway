# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import re
import types
import typing
from collections import abc as cabc

import tensordict as td
from torch import nn

from aioway.torch.guards import (
    is_op_overload_packet,
    is_tensor_method,
    is_torch_function,
)

from .guards import is_op_overload

__all__ = ["tcol_to_tdict", "render_torch_func_name", "camel_to_snake"]


def tcol_to_tdict(item) -> td.TensorDict:
    "Convert from tensor collection to `TensorDict`."

    if not td.is_tensor_collection(item):
        raise ValueError("Not tensor collection.")

    if isinstance(item, td.TensorDict):
        return item

    assert dcls.is_dataclass(item)
    attrs = dcls.asdict(item)
    result = td.from_dict(attrs)
    assert isinstance(result, td.TensorDict)
    return result


def render_torch_func_name(func: cabc.Callable[..., typing.Any]) -> str:
    name = func.__name__

    # Only descriptors use `__get__`, and we render the descriptor itself.
    if name == "__get__":
        assert isinstance(func, types.MethodType | types.MethodWrapperType), type(func)
        return repr(func.__self__)

    # It seems that there isn't an attribute that expose the name of the `OpOverload`,
    # so here we combine `namespace` (aten, prim, ...) and `__name__` (packet.type).
    if is_op_overload(func):
        return f"torch.ops.{func.namespace}.{name}"

    # Just converting to `str` works.
    if is_op_overload_packet(func):
        return f"torch.ops.{func!s}"

    # If it's `nn.Module`
    if isinstance(func, type) and issubclass(func, nn.Module):
        return f"nn.{name}"

    # If it's `torch.*`.
    if is_torch_function(func):
        return f"torch.{name}"

    # If it's `torch.Tensor.*`.
    if is_tensor_method(func):
        return f"torch.Tensor.{name}"

    # Don't know what this is. Just use `__qualname__`.
    return func.__qualname__


def camel_to_snake(name: str) -> str:
    return re.sub(_camel_case_regex(), "_", name).lower()


@functools.cache
def _camel_case_regex():
    return re.compile(r"(?<!^)(?=[A-Z])")
