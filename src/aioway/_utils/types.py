# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

__all__ = ["track_call_count", "Stack", "AnySet", "AnyDict"]


class _CallCounter[**P, T]:
    """
    Track the function's call count.
    The function appears to be the same in `repr`, improved `str`,
    and provide a new attribute `__invoke_count__` to track call count,
    which shows you the number of invokes performed on this funcion.
    You can access the original function in `__func__` attribute.
    """

    def __init__(self, func: cabc.Callable[P, T], /) -> None:
        self.__func = func
        self.__invokes = 0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        with self._increase_call_count():
            return self.__func(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self.__func)

    def __str__(self) -> str:
        string_builder = [
            self.__func.__module__,
            ".",
            self.__func.__qualname__,
            "[",
            str(self.__invoke_count__),
            "]",
            "(...)",
        ]
        return "".join(string_builder)

    def __get__(self, instance, owner):
        # If accessed from the class, return the wrapper itself
        if instance is None:
            return self

        @type(self)
        def bounded_method(*args, **kwargs):
            return self.__func__(instance, *args, **kwargs)

        return bounded_method

    @property
    def __func__(self) -> cabc.Callable[P, T]:
        "Returns the original funciton."
        return self.__func

    @property
    def __invoke_count__(self) -> int:
        "The number of times this function is being invoked."
        return self.__invokes

    @ctxl.contextmanager
    def _increase_call_count(self):
        try:
            self.__invokes += 1
            yield
        finally:
            self.__invokes -= 1


track_call_count = _CallCounter


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

    def append(self, obj: T, /) -> None:
        self.stack.append(obj)

    def pop(self) -> T:
        return self.stack.pop()

    def extend(self, it: cabc.Iterable[T], /) -> None:
        self.stack.extend(it)

    def truncate(self, after: int, /) -> None:
        "Truncate the items after the `after` index. Equivalent to `stack[after:]`."

        del self.stack[after:]

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


class AnySet[K = object]:
    """
    `AnySet` allows to store a set of items, using their `id`s to compare equality.
    """

    def __init__(self, base: type = object, /) -> None:
        self.__keys: dict[int, K] = {}
        """
        The keys that has been stored in the `AnyDict`.
        Using `dict` to avoid actually dereference `id`.
        """

        self.__type: type[K] = base
        "Store the type for `isinstance` checks."

    def __len__(self) -> int:
        return len(self.__keys)

    def __contains__(self, key: object, /) -> bool:
        if isinstance(key, self.__type):
            key_id = id(key)
            return key_id in self.__keys

        raise TypeError(f"{type(key)=} is not `{self.__type}`.")

    def __iter__(self) -> cabc.Iterator[K]:
        yield from self.__keys.values()

    def add(self, key: K) -> None:
        self.__keys[id(key)] = key

    def discard(self, key: K) -> None:
        if id(key) in self.__keys:
            del self.__keys[id(key)]


class AnyDict[K = object, V = object](AnySet[K]):
    """
    `AnyDict` allows you to treat `T` as if it's `Hashable` (it's not).
    Each item would be compared with `is` rather than `==`.
    """

    def __init__(self, base: type = object, /) -> None:
        super().__init__(base)

        self.__vals: dict[int, V] = {}
        """
        The values refered to by the `key`, using `id` as key.
        """

    def __getitem__(self, key: K, /) -> V:
        if key not in self:
            raise KeyError(f"{key=} is not found in `HopDict`.")

        return self.__vals[id(key)]

    def __setitem__(self, key: K, val: V, /) -> None:
        self.__assert_same_length()

        super().add(key)
        self.__vals[id(key)] = val

    def __delitem__(self, key: K, /) -> None:
        self.__assert_same_length()

        if key not in self:
            raise KeyError(f"{key=} is not in `HopDict`.")

        super().discard(key)
        del self.__vals[id(key)]

    def __assert_same_length(self):
        assert super().__len__() == len(self.__vals)
