# Copyright (c) AIoWay Authors - All Rights Reserved

"Module fwd/init modes, similar to `torch` function/dispatch modes."

import abc
import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._utils import track_call_count
from aioway.fn import Thunk, TorchThunk, thunk_dcls

from ._on_off import OnOffCtx, OnOffStack

__all__ = [
    "NnFwdFn",
    "NnInitFn",
    "NnFwdMode",
    "NnInitMode",
    "module_fwd",
    "module_init",
]

LOGGER = logging.getLogger(__name__)

FORWARDS: OnOffStack[NnFwdMode] = OnOffStack()
"`NnFwdMode` that is currently entered."

INITS: OnOffStack[NnInitMode] = OnOffStack()
"`NnInitMode` that is currently entered."


@track_call_count
def module_fwd(module: nn.Module, /, *args, **kwargs) -> typing.Any:
    """
    Call the `nn.Module`. This would execute all the modes at the outermost module.
    `aioway` functions must call this function to call `nn.Module`.

    For nested modules (fields of modules), use `register_*_hook` from `torch`.

    This function is recursive, so call counts are tracked to aid debugging.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected an `nn.Module` instance. Got {module=}.")

    return _invoke_rec(FORWARDS, NnFwdFn, module, args, kwargs)


@track_call_count
def module_init(init: type[nn.Module], /, *args, **kwargs) -> nn.Module:
    """
    Initialize the `nn.Module`. This would execute all the modes at the outermost module.
    `aioway` functions must call this function to initalize `nn.Module`.

    For nested modules (fields of modules), use `register_*_hook` from `torch`.

    This function is recursive, so call counts are tracked to aid debugging.
    """

    if not isinstance(init, type) or not issubclass(init, nn.Module):
        raise TypeError(
            f"Your module type {init} is not valid. Must be an `nn.Module` subclass."
        )

    result = _invoke_rec(INITS, NnInitFn, init, args, kwargs)

    if not isinstance(result, nn.Module):
        raise TypeError("Function `module_init` must return an `nn.Module`.")

    return result


def _invoke_rec[T: NnModeOnOff[typing.Any, typing.Any]](
    stack: OnOffStack[T],
    fn_type: type[TorchThunk[typing.Any]],
    call: cabc.Callable[..., typing.Any],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    """
    Essentially, invoke the given `call` recursively until the `stack` is exhausted.

    Overriding modes must only call `module_fwd` and `module_init`,
    which in turn calls this function to pop the next `mode` off the stack, and invoke it.

    This concept is borrowed from `__torch_function__` and `__torch_dispatch__`,
    you can see similarity when reading the code around their `_pop_mode_temporarily` function,
    which corresponds to our `borrow` function on the stack.
    """

    LOGGER.debug("Inovked on %s", stack)
    LOGGER.debug("type: %s", fn_type)
    LOGGER.debug("Thunk: %s", Thunk(call, *args, **kwargs))

    # Do not reinvoke the function! Call directly.
    if not stack:
        return call(*args, **kwargs)

    # Pop one `mode` for each call. At some point this would be exhausted.
    # And go to the previous `if not stack` shortcut.
    # For this to work, `mode(thunk)` must call `_invoke_rec` indirectly,
    # therefore you must call `module_fwd` / `module_init`,
    # or using `.do()` on `NnFwdFn` / `NnInitFn` does the same thing.
    with stack.borrow() as mode:

        if mode.on:
            thunk = fn_type(func=call, args=args, kwargs=kwargs)
            return mode(thunk)

        else:
            return _invoke_rec(stack, fn_type, call, args, kwargs)


@dcls.dataclass
class NnModeOnOff[T, V = object](OnOffCtx, abc.ABC):
    """
    The mixin for either `NnFwdMode`, `NnInitMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> V:
        raise NotImplementedError

    @typing.override
    @ctxl.contextmanager
    def enter(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        with self.STACK.hold(self):
            yield self


@typing.final
@thunk_dcls
class NnFwdFn(TorchThunk[nn.Module]):
    """
    `NnFwdFn` represents the module calls.

    It hooks into `module_fwd` so it allows for optional overwrites.
    """

    func: nn.Module
    "The module for the `Fn`."

    def __hash__(self) -> int:
        return id(self)

    @typing.override
    def do(self) -> object:
        return module_fwd(self.func, *self.args, **self.kwargs)

    def load_state_dict(
        self,
        state_dict: cabc.Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Expose the `nn.Module.load_state_dict` function to the users.
        """

        return self.func.load_state_dict(state_dict, strict=strict, assign=assign)

    def state_dict(self):
        """
        Expose the `nn.Module.state_dict` function to the users.
        """

        return self.func.state_dict()


class NnFwdMode(NnModeOnOff[NnFwdFn], abc.ABC):
    """
    `NnFwdMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.enter()` method (I prefer context managers).

    It is triggered when a `module_fwd` is called.
    """

    STACK = FORWARDS


@typing.final
@thunk_dcls
class NnInitFn(TorchThunk[type[nn.Module]]):
    """
    `NnInitFn` are used to initialize `nn.Module`s.

    It hooks into `module_init` so it allows for optional overwrites.
    """

    func: type[nn.Module]
    "The type of `nn.Module`."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")

    @typing.override
    def do(self) -> nn.Module:
        return module_init(self.func, *self.args, **self.kwargs)


class NnInitMode(NnModeOnOff[NnInitFn, nn.Module], abc.ABC):
    """
    `NnInitMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `.enter()` method (I prefer context managers).

    It is triggered when a `module_init` is called.
    """

    STACK = INITS
