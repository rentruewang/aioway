# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._pytrees import tree_leaves, tree_map_only
from aioway._utils import AnyDict, any_dict

__all__ = ["Op", "exec_op", "OpReplacer"]


@typing.dataclass_transform()
class _OpDclsMixin:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _ = dcls.dataclass()(cls)


class Op[T = object](_OpDclsMixin, abc.ABC):
    """
    The base class for expressions.

    Opessions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __call__(self) -> T:
        from .execs import executor

        return executor()(self)

    @property
    def __deps__(self) -> tuple[Op, ...]:
        """
        The `Op`s that this current expression depends on.
        This is found automatically by default.
        """

        def gen():
            for fo in _field_ops(self):
                yield from fo.ops

        return tuple(gen())

    @abc.abstractmethod
    def execute(self) -> T:
        """
        Evaluate and compute the result of this expression.
        """

        raise NotImplementedError

    def replace(self, replacement: AnyDict[Op, Op], /) -> Op:
        return OpReplacer(mapping=replacement)(self)

    def post_order(self) -> cabc.Iterator[Op]:
        """
        Get the post order traversal of `Op`.
        """

        for dep in self.__deps__:
            yield from dep.post_order()

        yield self


@dcls.dataclass(frozen=True)
class OpReplacer:
    """
    Replaces the `Op`s in `mapping`.

    Only `Op`s depending on a replaced `Op` are rebuilt (with `dcls.replace`),
    the rest are kept as is. Each `Op` is visited at most once.
    """

    mapping: AnyDict[Op, Op]
    "The `Op`s to replace, and their replacements."

    seen: AnyDict[Op, Op] = dcls.field(default_factory=any_dict)
    "The result for each `Op` visited so far."

    def __call__(self, op: Op) -> Op:
        if op not in self.seen:
            self.seen[op] = self._maybe_replace(op)

        return self.seen[op]

    def _maybe_replace(self, op: Op) -> Op:
        if op in self.mapping:
            return self.mapping[op]

        changes: dict[str, typing.Any] = {}
        for old in _field_ops(op):
            # Convert also nested `Op`s in containers.
            new = _FieldOps(old.name, tree_map_only(Op, self, old.value))

            # Replacement has happened.
            olds, news = old.ops, new.ops
            if len(olds) != len(news) or any(o is not n for o, n in zip(olds, news)):
                changes[new.name] = new.value

        # All children unchanged, and `op` itself is not replaced.
        if not changes:
            return op

        return dcls.replace(op, **changes)


@dcls.dataclass(frozen=True)
class _FieldOps:
    name: str
    "The field name."

    value: typing.Any
    "The field value."

    @property
    def ops(self) -> list[Op]:
        "The `Op`s nested in `value`, found by pytree."

        def leaves():
            is_op = lambda f: isinstance(f, Op)
            for leaf in tree_leaves(self.value, is_leaf=is_op):
                if isinstance(leaf, Op):
                    yield leaf

        return list(leaves())


def _field_ops(op: Op) -> cabc.Iterator[_FieldOps]:
    for field in dcls.fields(op):
        value = getattr(op, field.name)
        yield _FieldOps(field.name, value)


def exec_op(obj):
    """
    Execute `obj` if it is `Op`, else return identity.
    """

    if isinstance(obj, Op):
        return obj()

    else:
        return obj
