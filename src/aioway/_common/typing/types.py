# Copyright (c) AIoWay Authors - All Rights Reserved

"Render the class hierarchies."

import typing
from collections import abc as cabc

import rich
from rich import syntax, tree

__all__ = ["subclass_tree", "print_subclass_tree", "render_class_syntax"]


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
