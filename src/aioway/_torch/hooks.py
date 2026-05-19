# Copyright (c) AIoWay Authors - All Rights Reserved

"The context manager version of hooks in `torch.nn.modules.module` and `nn.Module`."

import contextlib as ctxl
import typing

from torch import nn
from torch.nn.modules import module as _M

__all__ = [
    "ModulePreHook",
    "ModuleHook",
    "register_module_forward_pre_hook",
    "register_module_forward_hook",
    "register_module_full_backward_pre_hook",
    "register_module_full_backward_hook",
    "nn_module_register_forward_pre_hook",
    "nn_module_register_forward_hook",
    "nn_module_register_full_backward_pre_hook",
    "nn_module_register_full_backward_hook",
]


class ModulePreHook(typing.Protocol):
    def __call__(self, module: nn.Module, input) -> typing.Any: ...


class ModuleHook(typing.Protocol):
    def __call__(self, module: nn.Module, input, output) -> typing.Any: ...


@ctxl.contextmanager
def register_module_forward_pre_hook(hook: ModulePreHook):
    """
    The context manager version of global `register_module_forward_pre_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `register_module_forward_pre_hook` does.
    """

    handle = _M.register_module_forward_pre_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def register_module_forward_hook(hook: ModuleHook, *, always_call: bool = False):
    """
    The context manager version of global `register_module_forward_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `register_module_forward_hook` does.
    """

    handle = _M.register_module_forward_hook(hook, always_call=always_call)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def register_module_full_backward_pre_hook(hook: ModulePreHook):
    """
    The context manager version of global `register_module_full_backward_pre_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `register_module_full_backward_pre_hook` does.
    """

    handle = _M.register_module_full_backward_pre_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def register_module_full_backward_hook(hook: ModuleHook):
    """
    The context manager version of global `register_module_full_backward_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `register_module_full_backward_hook` does.
    """

    handle = _M.register_module_full_backward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def nn_module_register_forward_pre_hook(module: nn.Module, hook: ModulePreHook):
    """
    The context manager version of `nn.Module.register_forward_pre_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `nn.Module.register_forward_pre_hook` does.
    """

    handle = module.register_forward_pre_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def nn_module_register_forward_hook(
    module: nn.Module, hook: ModuleHook, *, always_call: bool = False
):
    """
    The context manager version of `nn.Module.register_forward_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `nn.Module.register_forward_hook` does.
    """

    handle = module.register_forward_hook(hook, always_call=always_call)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def nn_module_register_full_backward_pre_hook(module: nn.Module, hook: ModulePreHook):
    """
    The context manager version of `nn.Module.register_full_backward_pre_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `nn.Module.register_full_backward_pre_hook` does.
    """

    handle = module.register_full_backward_pre_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()


@ctxl.contextmanager
def nn_module_register_full_backward_hook(module: nn.Module, hook: ModuleHook):
    """
    The context manager version of `nn.Module.register_full_backward_hook`,
    with the hook removed automatically outside of the scope.
    See `torch` documentation for `nn.Module.register_full_backward_hook` does.
    """

    handle = module.register_full_backward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()
