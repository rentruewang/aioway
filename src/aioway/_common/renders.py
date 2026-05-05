# Copyright (c) AIoWay Authors - All Rights Reserved

"Print the objects to the terminal in a nice way."

import typing
from collections import abc as cabc

import rich
from rich import syntax, tree

__all__ = [
    "render_fcall",
    "subclass_tree",
    "print_subclass_tree",
    "render_class_syntax",
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
