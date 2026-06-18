# Copyright (c) AIoWay Authors - All Rights Reserved

"Module fwd/init modes, similar to `torch` function/dispatch modes."

import abc
import logging
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._thunks import AnyThunk, TorchThunk
from aioway._utils import render_fcall, render_torch_func_name, track_call_count

from .modes import Mode, ModeStack

__all__ = ["NnFwdThunk", "NnInitThunk", "NnFwdMode", "NnInitMode"]

LOGGER = logging.getLogger(__name__)

FORWARDS: ModeStack[NnFwdMode] = ModeStack()
"`NnFwdMode` that is currently entered."

INITS: ModeStack[NnInitMode] = ModeStack()
"`NnInitMode` that is currently entered."


@typing.final
class NnFwdThunk(TorchThunk):
    """
    `NnFwdThunk` represents the module calls.

    It works like `__torch_function__`, pops the context from the global stack,
    execute the `.run` function (which might trigger recursive calls), then push it back.
    """

    if typing.TYPE_CHECKING:

        @property
        def func(self) -> nn.Module: ...

    def __init__(self, func: nn.Module, *args, **kwargs) -> None:
        super().__init__(func, *args, **kwargs)

        if not isinstance(self.func, nn.Module):
            raise TypeError(f"Expected an `nn.Module`, got {type(self.func)=}.")

    def __repr__(self):
        return render_fcall(self.func, *self.args, **self.kwargs)

    def __hash__(self) -> int:
        return id(self)

    @track_call_count
    def __call__(self) -> object:
        """
        Call the `nn.Module`. This would execute all the modes at the outermost module.
        `aioway` functions must call this function to call `nn.Module`.

        For nested modules (fields of modules), use `register_*_hook` from `torch`.

        This function is recursive, so call counts are tracked to aid debugging.
        """

        return _invoke_rec(FORWARDS, NnFwdThunk, self.func, self.args, self.kwargs)

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


class NnFwdMode(Mode[NnFwdThunk, object], abc.ABC):
    """
    `NnFwdMode` is the mode for similar to `__torch_function__` / `__torch_dispatch__`,
    except you enter / exit with a `Mode.__call__()` method (I prefer context managers).

    It is triggered when a `NnFwdThunk.__call__` is called.
    """

    STACK = FORWARDS


@typing.final
class NnInitThunk[**P = ...](TorchThunk):
    """
    `NnInitThunk` are used to initialize `nn.Module`s.

    It works like `__torch_function__`, pops the context from the global stack,
    execute the `.run` function (which might trigger recursive calls), then push it back.
    """

    if typing.TYPE_CHECKING:

        @property
        def func(self) -> nn.Module: ...

    def __init__(
        self, func: cabc.Callable[P, nn.Module], *args: P.args, **kwargs: P.kwargs
    ) -> None:
        super().__init__(func, *args, **kwargs)

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")

    def __repr__(self):
        func_name = render_torch_func_name(self.func)
        return render_fcall(f"nn_init::{func_name}", *self.args, **self.kwargs)

    @track_call_count
    def __call__(self) -> nn.Module:
        """
        Initialize the `nn.Module`. This would execute all the modes at the outermost module.
        `aioway` functions must call this function to initalize `nn.Module`.

        For nested modules (fields of modules), use `register_*_hook` from `torch`.

        This function is recursive, so call counts are tracked to aid debugging.
        """

        result = _invoke_rec(INITS, NnInitThunk, self.func, self.args, self.kwargs)

        if not isinstance(result, nn.Module):
            raise TypeError("Function `module_init` must return an `nn.Module`.")

        return result


class NnInitMode(Mode[NnInitThunk, nn.Module], abc.ABC):
    """
    `NnInitMode` is a `Mode` for `nn.Module.__init__`.

    It is triggered when a `NnInitThunk.__call__` is called.
    """

    STACK = INITS


def _invoke_rec[T: Mode[typing.Any, typing.Any]](
    stack: ModeStack[T],
    fn_type: type[TorchThunk],
    call: cabc.Callable[..., typing.Any],
    args: tuple[typing.Any, ...],
    kwargs: dict[str, typing.Any],
):
    """
    Essentially, invoke the given `call` recursively until the `stack` is exhausted.

    Overriding modes must only call `NnFwdThunk.__call__` and `NnInitThunk.__call__`,
    which in turn calls this function to pop the next `mode` off the stack, and invoke it.

    This concept is borrowed from `__torch_function__` and `__torch_dispatch__`,
    you can see similarity when reading the code around their `_pop_mode_temporarily` function,
    which corresponds to our `borrow` function on the stack.
    """

    LOGGER.debug("Inovked on %s", stack)
    LOGGER.debug("type: %s", fn_type)
    LOGGER.debug("AnyThunk: %s", AnyThunk(call, *args, **kwargs))

    # Do not reinvoke the function! Call directly.
    if not stack:
        return call(*args, **kwargs)

    # Pop one `mode` for each call. At some point this would be exhausted.
    # And go to the previous `if not stack` shortcut.
    # For this to work, `mode(thunk)` must call `_invoke_rec` indirectly,
    # therefore you must call `module_fwd` / `module_init`,
    # or using `.run()` on `NnFwdThunk` / `NnInitThunk` does the same thing.
    with stack.borrow() as mode:

        if mode.on:
            thunk = fn_type(call, *args, **kwargs)
            return mode.run(thunk)

        else:
            return _invoke_rec(stack, fn_type, call, args, kwargs)
