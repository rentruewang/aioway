# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections import abc as cabc

from aioway._pytrees import tree_leaves, tree_map_only
from aioway._utils import AnyDict, Stack, any_dict
from aioway.t import Domain

__all__ = ["Op", "OpDomainMismatch", "OpReplacer"]


_callers: Stack[Op] = Stack()
"The `Op`s whose `__call__` is running, the last is the current `Op`."


class OpDomainMismatch(ValueError):
    "The exception is raised by `__domain__` to signal `Op` is invalid."


@typing.dataclass_transform()
class _OpDclsMixin:
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _ = dcls.dataclass()(cls)


class Op[T = typing.Any](_OpDclsMixin, abc.ABC):
    """
    The base class for expressions.

    Opessions refer to those operations that are

    1. Pure.
    2. Operate on batches.

    It can be converted into an `nn.Module` easily.
    """

    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __post_init__(self) -> None:
        # Instantiate the domain, also performs checks.
        _ = self.__domain__

    @property
    def __domain__(self) -> Domain:
        self._static_check_domain()
        return self._infer_domain()

    @property
    @abc.abstractmethod
    def __domain_deps__(self) -> list[Domain]:
        """
        The domains required of `__deps__`, the same length and order.
        """

        raise NotImplementedError

    def _static_check_domain(self) -> None:
        """
        Check the domains of `__deps__` against `__domain_deps__`.

        A sub domain is a match.
        A super domain (e.g. `ANY`) may match, checked by `_input` on run time.

        Raises:
            OpDomainMismatch: if neither a sub domain nor a super domain.
        """

        deps, required = self.__deps__, self.__domain_deps__

        if len(deps) != len(required):
            raise OpDomainMismatch(f"{len(deps)=} != {len(required)=}")

        for dep, req in zip(deps, required):
            actual = dep.__domain__

            # For `<=` case, it's legal.
            # For `>` case, need a run time check, so, not now.
            if actual <= req or actual > req:
                continue

            raise OpDomainMismatch(f"expected {req}, got {actual}")

    @abc.abstractmethod
    def _infer_domain(self) -> Domain:
        """
        The output domain, given that `_static_check_domain` passed.
        """

        raise NotImplementedError

    def _get_dep_value(self, dep: Op) -> typing.Any:
        """
        Execute `dep`, one of `__deps__`.

        If its domain is a super domain of the requirement,
        the result is checked here, on run time.
        """

        value = dep()

        # By identity, as `==` on `Op`s compares fields.
        req = next(
            domain
            for domain, op in zip(self.__domain_deps__, self.__deps__)
            if op is dep
        )

        # Requested type is a subtype of the type given, check.
        if dep.__domain__ > req:
            req.check(value)

        return value

    def __call__(self) -> T:
        from .execs import executor

        _callers.append(self)
        try:
            result = executor()(self)
        finally:
            _callers.pop()

        self._domain_checks(result)
        return result

    def _domain_checks(self, result) -> None:
        # Check output.
        domain = self.__domain__
        domain.check(result)

        # No caller of `self`.
        if not _callers:
            return

        caller = _callers.top()

        # No check needed.
        if domain <= (required := self._get_domain_in_caller(caller)):
            return

        # `Domain` of `self` is too broad, runtime assert needed.
        if domain > required:
            required.check(result)
            return

        raise OpDomainMismatch("Incompatible domains.")

    def _get_domain_in_caller(self, caller: Op) -> Domain:
        # Get `self` in the `caller`'s `__deps__` and corresponding `Domain`.
        for dom, op in zip(caller.__domain_deps__, caller.__deps__):
            if op is self:
                return dom

        raise RuntimeError("Cannot find `self` in `caller`, impossible.")

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
    def _execute(self) -> T:
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
