# Copyright (c) AIoWay Authors - All Rights Reserved

"A bunch of context managers controlling the fake mode."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import tensordict as td
import torch
from torch import _subclasses as tsc

__all__ = [
    "fake_mode",
    "fake_mode_func",
    "is_fake_tensor",
    "is_real_tensor",
    "to_fake_tensor",
    "to_fake_tensordict",
    "enabled_fake_mode",
]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class _FakeModeRc:
    """
    Do "reference counting" for fake mode.
    """

    mode: tsc.FakeTensorMode
    "The fake mode instance that shall be entered."

    count: int = 0
    "The enter count."

    def __bool__(self) -> bool:
        return bool(self.active())

    @ctxl.contextmanager
    def __call__(self):
        with self.mode, self._count_mode():
            yield self.mode

    @ctxl.contextmanager
    def _count_mode(self):
        try:
            self.count += 1
            yield
        finally:
            self.count -= 1

    def active(self) -> bool:
        return self.count != 0


_FAKE_MODE = tsc.FakeTensorMode(allow_non_fake_inputs=True)
_FAKE_MODE_RC = _FakeModeRc(_FAKE_MODE)


def to_fake_tensor(tensor: torch.Tensor) -> tsc.FakeTensor:
    """
    Move a possibly real tensor to a fake torch.Tensor
    """

    if is_fake_tensor(tensor):
        return tensor

    with fake_mode() as mode:
        converter = mode.fake_tensor_converter
        return converter.from_real_tensor(mode, tensor)


def to_fake_tensordict(tensordict: td.TensorDict) -> td.TensorDict:
    result = td.TensorDict(
        {key: to_fake_tensor(val) for key, val in tensordict.items()}
    )
    result.shape = tensordict.shape
    return result


def is_real_tensor(tensor: object) -> typing.TypeIs[torch.Tensor]:
    """
    Detect if a tensor is a normal tensor.
    """

    return isinstance(tensor, torch.Tensor) and not is_fake_tensor(tensor)


def is_fake_tensor(tensor: object) -> typing.TypeIs[tsc.FakeTensor]:
    """
    Detect if a tensor is a fake tensor.
    """

    return isinstance(tensor, tsc.FakeTensor)


def enabled_fake_mode() -> tsc.FakeTensorMode | None:
    """
    Get the current fake mode, is available.

    This can be used in an `if` or a `with`.
    """

    if _FAKE_MODE_RC.active():
        return _FAKE_MODE_RC.mode
    else:
        return None


@ctxl.contextmanager
def fake_mode():
    """
    Enable `torch`'s fake mode s.t. we can do symbolic processing easily.

    Since fake mode doesn't nest (it seems), if fake mode is already on, yield that.
    """

    with _FAKE_MODE_RC():
        yield _FAKE_MODE_RC.mode


def fake_mode_func[**P, T](func: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
    """
    Decorator on a function, s.t. when the function is being called, fake mode is enabled.
    """

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with fake_mode():
            return func(*args, **kwargs)

    wrapper.__qualname__ = func.__qualname__
    wrapper.__doc__ = func.__doc__

    return wrapper
