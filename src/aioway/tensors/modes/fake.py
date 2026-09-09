# Copyright (c) AIoWay Authors - All Rights Reserved

"A bunch of context managers controlling the fake mode."

import contextlib as ctxl
from collections import abc as cabc

from torch._subclasses import fake_tensor as ft

__all__ = [
    "fake_mode",
    "real_mode",
    "torch_set_fake_mode",
    "torch_set_fake_mode_func",
    "is_fake_mode_on",
    "active_fake_mode",
]


_FAKE_MODE = ft.FakeTensorMode(allow_non_fake_inputs=True)
_fake_mode_is_active: bool = False


def is_fake_mode_on() -> bool:
    """
    Check if we are running under a `fake_mode` context.
    """

    return active_fake_mode() is not None


def active_fake_mode() -> ft.FakeTensorMode | None:
    """
    Get the fake mode if it is active, or `None` if no fake mode is active.
    """

    if _fake_mode_is_active:
        return _FAKE_MODE
    else:
        return None


@ctxl.contextmanager
def fake_mode():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    with _FAKE_MODE, _set_fake_mode_flag(True):
        yield _FAKE_MODE


@ctxl.contextmanager
def real_mode():
    """
    Disable `torch`'s fake mode temporarily.

    Yields:
        The context manager that is pushed to torch's dispatch stack.
    """

    with ft.unset_fake_temporarily() as mode, _set_fake_mode_flag(False):
        yield mode


def torch_set_fake_mode(yes: bool, /):
    """
    Context manager to set the fake mode if `True` or `False` to set to the real mode.
    """

    if yes:
        return fake_mode()
    else:
        return real_mode()


@ctxl.contextmanager
def _set_fake_mode_flag(to: bool):
    global _fake_mode_is_active

    before = _fake_mode_is_active
    _fake_mode_is_active = to
    try:
        yield _fake_mode_is_active
    finally:
        _fake_mode_is_active = before


def torch_set_fake_mode_func(to: bool, /):
    def decorator[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
        """
        Decorator on a function, s.t. when the function is being called, fake mode is enabled.
        """

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with torch_set_fake_mode(to):
                return func(*args, **kwargs)

        _set_wrapper_func(wrapper, func)
        return wrapper

    return decorator


def _set_wrapper_func[**P, T](
    wrapper: cabc.Callable[P, T], func: cabc.Callable[P, T]
) -> None:
    wrapper.__qualname__ = func.__qualname__
    wrapper.__name__ = func.__name__
    wrapper.__module__ = func.__module__
    wrapper.__doc__ = func.__doc__
