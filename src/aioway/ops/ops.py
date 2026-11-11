# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._utils import AnyDict, any_dict

__all__ = ["Op", "exec_op"]


@typing.dataclass_transform()
class _OpDclsMixin:

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()
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

    @abc.abstractmethod
    def execute(self) -> T:
        """
        Evaluate and compute the result of this expression.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def deps(self) -> cabc.Iterable[Op]:
        """
        The expressions that this current expression depends on.
        """

        raise NotImplementedError

    def replace(self, replacement: AnyDict[Op, Op], /) -> typing.Self:
        result = OpReplacer(mapping=replacement)(self)
        assert type(result) == type(self)
        return result

    def post_order(self) -> cabc.Iterator[Op]:
        """
        Get the post order traversal of `Op`.
        """

        for dep in self.deps():
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

        changes: dict[str, Op] = {}
        for field in dcls.fields(op):
            val = getattr(op, field.name)
            if not isinstance(val, Op):
                continue

            # Replacement has happened.
            if (new := self(val)) is not val:
                changes[field.name] = new

        # All children unchanged, and `op` itself is not replaced.
        if not changes:
            return op

        return dcls.replace(op, **changes)


def exec_op(obj):
    """
    Execute `obj` if it is `Op`, else return identity.
    """

    if isinstance(obj, Op):
        return obj()

    else:
        return obj
