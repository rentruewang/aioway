# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = [
    "dcls_no_eq",
    "dcls_no_repr",
    "dcls_frozen_no_repr",
    "dcls_frozen_slots",
    "dcls_no_eq_no_repr",
    "Stack",
]


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False)(cls)
    return result


@typing.dataclass_transform(eq_default=False, frozen_default=True)
def dcls_frozen_slots[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(frozen=True, slots=True)(cls)
    return result


@typing.dataclass_transform(eq_default=True)
def dcls_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(repr=False)(cls)
    return result


@typing.dataclass_transform(eq_default=True, frozen_default=True)
def dcls_frozen_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(repr=False, frozen=True)(cls)
    return result


@typing.dataclass_transform(eq_default=False)
def dcls_no_eq_no_repr[T: type](cls: T) -> T:
    result: typing.Any = dcls.dataclass(eq=False, repr=False)(cls)
    return result


@dcls.dataclass(frozen=True)
class Stack[T]:
    """
    `Stack` is a scope tracker for s.t. it's easier to monitor in terms of crashes.
    """

    stack: list[T] = dcls.field(default_factory=list)
    """
    The stack that is currently in scope.
    """

    def __bool__(self) -> bool:
        return bool(len(self))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return len(self.stack)

    @typing.overload
    def __getitem__(self, idx: int) -> T: ...

    @typing.overload
    def __getitem__(self, idx: slice[int]) -> typing.Self: ...

    def __getitem__(self, idx: int | slice[int]):
        match idx:
            case int():
                return self.stack[idx]
            case slice():
                return type(self)(self.stack[idx])

        raise TypeError(type(idx))

    def top(self) -> T:
        return self.stack[-1]

    def append(self, fn: T) -> None:
        self.stack.append(fn)

    def pop(self) -> T:
        return self.stack.pop()

    @ctxl.contextmanager
    def hold(self, item: T):
        """
        Enter the scope, push the `item` onto the stack, and then pop when exit.
        """

        self.append(item)
        try:
            yield
        finally:
            _ = self.pop()

    @ctxl.contextmanager
    def borrow(self) -> cabc.Generator[T]:
        """
        Temporarily pop the last item, then push it back after exiting the scope.

        Yields:
            The last item (which would be pushed back later).
        """

        item = self.stack.pop()
        try:
            yield item
        finally:
            self.append(item)
