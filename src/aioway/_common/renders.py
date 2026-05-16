# Copyright (c) AIoWay Authors - All Rights Reserved

"Print the objects to the terminal in a nice way."

import types
import typing
from collections import abc as cabc

import rich
import torch
from rich import syntax, tree
from torch import _ops

from .decomps import replace_tensors_with_attr

__all__ = [
    "render_fcall",
    "subclass_tree",
    "print_subclass_tree",
    "render_class_syntax",
    "render_func_name",
    "render_tensor_func_short",
]

type FunctionLike = str | cabc.Callable[..., typing.Any]


def render_fcall(func: FunctionLike, *args: typing.Any, **kwargs: typing.Any) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    return f"{func!s}({args_str})"


def render_class_syntax(cls: type):
    return syntax.Syntax(f"class {cls.__module__}.{cls.__qualname__}", lexer="py")


def render_tensor_func_short(func: str, args, kwargs) -> str:
    # `Attr`s are better for display than `torch.Tensor`s.

    args = replace_tensors_with_attr(args)
    kwargs = replace_tensors_with_attr(kwargs)

    return render_fcall(func, *args, **kwargs)


def subclass_tree(cls: type, render: cabc.Callable[[type], typing.Any] = repr):
    t = tree.Tree(render(cls))

    _subclass_tree(cls, t, seen=set(), render=render)

    return t


def print_subclass_tree(
    cls: type, render: cabc.Callable[[type], typing.Any] = render_class_syntax
) -> None:
    """
    Print the class and render it with `rich.syntax.Syntax`.
    """
    rich.print(subclass_tree(cls, render=render))


def _subclass_tree(
    cls: type,
    tree: tree.Tree,
    seen: set[type],
    render: cabc.Callable[[type], typing.Any],
):
    """
    Convert subclass into a `rich.tree.Tree`.
    """

    if cls in seen:
        return

    seen.add(cls)

    for sub_cls in cls.__subclasses__():
        sub_tree = tree.add(render(sub_cls))
        _subclass_tree(sub_cls, sub_tree, seen=seen, render=render)


def render_func_name(func: cabc.Callable[..., typing.Any]) -> str:
    name = func.__name__

    # Only descriptors use `__get__`, and we render the descriptor itself.
    if name == "__get__":
        assert isinstance(func, types.MethodType | types.MethodWrapperType), type(func)
        return repr(func.__self__)

    # It seems that there isn't an attribute that expose the name of the `OpOverload`,
    # so here we combine `namespace` (aten, prim, ...) and `__name__` (packet.type).
    if isinstance(func, _ops.OpOverload):
        return f"torch.ops.{func.namespace}.{name}"

    # Just converting to `str` works.
    if isinstance(func, _ops.OpOverloadPacket):
        return f"torch.ops.{func!s}"

    # If it's `torch.*`.
    if getattr(torch, name, None) is func:
        return f"torch.{name}"

    # If it's `torch.Tensor.*`.
    if getattr(torch.Tensor, name, None) is func:
        return f"torch.Tensor.{name}"

    # Don't know what this is. Just use `__qualname__`.
    return func.__qualname__
