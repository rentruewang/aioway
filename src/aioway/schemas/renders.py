# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

from aioway._utils import render_fcall, render_torch_func_name

from .utils import replace_tensors_with_attr

__all__ = ["render_tensor_func_short", "render_function_body_prefix"]


def render_tensor_func_short(func: str, args, kwargs) -> str:
    # `Attr`s are better for display than `torch.Tensor`s.

    args = replace_tensors_with_attr(args)
    kwargs = replace_tensors_with_attr(kwargs)

    return render_fcall(func, *args, **kwargs)


def render_function_body_prefix(
    prefix: str,
    func: cabc.Callable[..., typing.Any],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
) -> str:
    func_name = render_torch_func_name(func)
    return render_tensor_func_short(prefix + "::" + func_name, args, kwargs)
