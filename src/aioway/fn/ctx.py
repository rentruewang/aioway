# Copyright (c) AIoWay Authors - All Rights Reserved

"A bunch of context managers controlling the fake mode."

import contextlib as ctxl
import logging
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch._subclasses import fake_tensor as ft

__all__ = [
    "torch_fake_mode",
    "torch_enable_fake_mode",
    "torch_enable_fake_mode_func",
    "is_fake_tensor",
    "is_real_tensor",
    "to_fake_tensor",
    "to_fake_tensordict",
    "enabled_fake_mode",
    "torch_real_mode",
    "torch_disable_torch_func",
]

LOGGER = logging.getLogger(__name__)


_FAKE_MODE = ft.FakeTensorMode(allow_non_fake_inputs=True)
_fake_mode_is_active: bool = False


def to_fake_tensor(tensor: torch.Tensor) -> ft.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with torch_fake_mode() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def to_fake_tensordict(tdict: td.TensorDict) -> td.TensorDict:
    result = td.TensorDict({key: to_fake_tensor(val) for key, val in tdict.items()})
    result.shape = tdict.shape
    return result


def is_real_tensor(tensor: object) -> typing.TypeIs[torch.Tensor]:
    """
    Detect if a tensor is a normal tensor.
    """

    return isinstance(tensor, torch.Tensor) and not is_fake_tensor(tensor)


def is_fake_tensor(tensor: object) -> typing.TypeIs[ft.FakeTensor]:
    """
    Detect if a tensor is a fake tensor.
    """

    return isinstance(tensor, ft.FakeTensor)


def enabled_fake_mode() -> ft.FakeTensorMode | None:
    """
    Get the current fake mode, is available.

    This can be used in an `if` or a `with`.
    """

    if _fake_mode_is_active:
        return _FAKE_MODE
    else:
        return None


@ctxl.contextmanager
def torch_fake_mode():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    with _FAKE_MODE, _set_fake_mode_flag(True):
        yield _FAKE_MODE


@ctxl.contextmanager
def torch_real_mode():
    """
    Disable `torch`'s fake mode temporarily.

    Yields the context manager that is pushed to torch's dispatch stack.
    """

    with ft.unset_fake_temporarily() as mode, _set_fake_mode_flag(False):
        yield mode


def torch_enable_fake_mode(yes: bool, /):
    """
    Context manager to set the fake mode if `True` or `False` to set to the real mode.
    """

    if yes:
        return torch_fake_mode()
    else:
        return torch_real_mode()


@ctxl.contextmanager
def _set_fake_mode_flag(to: bool):
    global _fake_mode_is_active
    before = _fake_mode_is_active

    _fake_mode_is_active = to

    try:
        yield
    finally:
        _fake_mode_is_active = before


def torch_disable_torch_func[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
    """
    Disable `__torch_function__` mode, for the wrapped function.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with torch.DisableTorchFunction():
            return func(*args, **kwargs)

    _set_wrapper_func(wrapper, func)
    return wrapper


def torch_enable_fake_mode_func(to: bool, /):
    def decorator[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
        """
        Decorator on a function, s.t. when the function is being called, fake mode is enabled.
        """

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with torch_enable_fake_mode(to):
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
