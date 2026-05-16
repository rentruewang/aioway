# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Fn`s corresponding to module's."

import abc
import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway.fn.fn import OnOffStack
from aioway.might import Might, find_might

from .fn import OnOffCtx, OnOffStack, TorchThunk

__all__ = ["NnFwdFn", "NnInitFn", "MightFn"]

_FORWARDS: OnOffStack[MFwdMode] = OnOffStack()
"`MFwdMode` that is currently entered."

_INITS: OnOffStack[MInitMode] = OnOffStack()
"`MInitMode` that is currently entered."


@dcls.dataclass
class MModeOnOff[T](OnOffCtx, abc.ABC):
    """
    The mixin for either `MFwdMode`, `MInitMode`.
    """

    @abc.abstractmethod
    def __call__(self, thunk: T, /) -> object:
        raise NotImplementedError

    @typing.override
    @ctxl.contextmanager
    def ctx(self: typing.Self):
        """
        Enter the `__torch_function__` / `__torch_dispatch__` context,
        and store the mode itself s.t. it can be turned on / off later.
        """

        if not self.on:
            yield self
            return

        with self.STACK.enter(self):
            yield self


@dcls.dataclass
class NnFwdFn(TorchThunk[nn.Module]):
    "`NnFwdFn` represents the module calls."

    func: nn.Module
    "The module for the `Fn`."

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


class MFwdMode(MModeOnOff[NnFwdFn], abc.ABC):
    STACK = _FORWARDS


class NnInitFn(TorchThunk[type[nn.Module]]):
    """
    `NnInitFn` is the "leftover" `nn.Module`s that are not covered by the `Might` API.
    """

    func: type[nn.Module]
    "The type of `nn.Module`."

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.func, type) or not issubclass(self.func, nn.Module):
            raise TypeError(f"{self.func} should be a subclass of `nn.Module`.")


class MInitMode(MModeOnOff[NnInitFn], abc.ABC):
    STACK = _INITS


@dcls.dataclass
class MightFn:
    """
    `MightFn` are `Fn` that wrap `Might`s, which are supported `nn.Module` ops.
    """

    might: Might
    "The `Might` instance."

    def do(self) -> object:
        return self.might.do()

    @classmethod
    def find_might(cls, thunk: NnInitFn) -> typing.Self:
        might = find_might(thunk.func, *thunk.args, **thunk.kwargs)

        if might is NotImplemented:
            return NotImplemented

        else:
            return cls(might)
