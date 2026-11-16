# Copyright (c) AIoWay Authors - All Rights Reserved

"""
Tracers that record torch calls as an `Op` graph.

Each recorded call becomes a `CallOp` / `ModuleThunkOp`,
whose tensor arguments are replaced by the `Op`s that produced them.

Tensors that no recorded call produced (the inputs) are held by `TensorOp`s,
and become the parameters `t0`, `t1`, ... of `OpHist.exec`.
"""

import abc
import contextlib as ctxl
import dataclasses as dcls
import operator
import typing
from collections import abc as cabc

import torch
from torch import nn

from aioway._pytrees import (
    GetAttrKey,
    KeyPath,
    MappingKey,
    SequenceKey,
    key_get,
    tree_leaves_with_path,
    tree_map_only,
)
from aioway._signs import Param
from aioway._utils import AnyDict, Stack, any_dict
from aioway.t import (
    TensorDomain,
    TorchDispMode,
    TorchFuncMode,
    fake_mode,
    is_aten_op,
    is_tensor_method,
    is_torch_function,
    register_module_forward_hook,
    register_module_forward_pre_hook,
    route_aten_mode,
)

from .funcs import FuncInputOp, OpFunc
from .ops import Op
from .thunks import CallOp, ModuleThunkOp
from .vars import TensorOp

__all__ = [
    "OpHist",
    "TorchFuncOpDag",
    "fake_aten_op_dag",
    "ModuleOpTracker",
    "ModuleOpHist",
    "track_module_ops",
]


# The history ====


@dcls.dataclass
class OpHist:
    """
    The history of calls, linked by tensors into an `Op` graph.

    Shared by the torch call tracers and the module tracer.
    """

    history: list[CallOp] = dcls.field(default_factory=list)
    """
    The history encountered.
    """

    output_index: AnyDict[torch.Tensor, Op] = dcls.field(default_factory=any_dict)
    """
    The `Op` of each tensor where `thunk.output = tensor`.

    Since this graph is about expression, we only care about the output.
    """

    input_index: AnyDict[torch.Tensor, TensorOp] = dcls.field(default_factory=any_dict)
    """
    The `TensorOp` placeholder of each tensor used but not produced,
    in order of first use.
    """

    def __len__(self) -> int:
        return len(self.history)

    def __getitem__(self, idx: int) -> CallOp:
        return self.history[idx]

    def lift[O: CallOp](
        self,
        cls: type[O],
        func: cabc.Callable,
        args: tuple[typing.Any, ...],
        kwargs: dict[str, typing.Any],
    ) -> O:
        "Build the `cls` thunk, with tensors in the arguments replaced by `Op`s."

        return cls(
            func=func,
            args=tree_map_only(torch.Tensor, self._op_of, args),
            kwargs=tree_map_only(torch.Tensor, self._op_of, kwargs),
        )

    def append(self, thunk: CallOp, result: typing.Any) -> None:
        outputs = _result_ops(thunk, result)

        # Check if the keys already exists,
        # should be unique due to cloning in fake mode.
        if not self.output_index.keys().isdisjoint(outputs.keys()):
            raise KeyError(f"Impossible conflicting output at {thunk}.")

        self.history.append(thunk)

        for output, op in outputs.items():
            self.output_index[output] = op

    def thunk_of(self, output: torch.Tensor) -> Op:
        return self.output_index[output]

    @property
    def output(self) -> Op:
        "The last thunk, whose result is returned by `exec`."

        if not self.history:
            raise ValueError("Nothing is recorded.")

        return self.history[-1]

    @property
    def inputs(self) -> list[torch.Tensor]:
        "The traced tensors bound to `exec`'s parameters, in signature order."
        return [placeholder.tensor for placeholder in self._signature_inputs()]

    @property
    def exec(self) -> OpFunc:
        """
        The `OpFunc` of the graph, with unproduced tensors as parameters.

        Parameters are named `t0`, `t1`, ... in the order `OpFunc` binds them
        (post order from `output`), so names and positions always agree.
        """

        mapping: AnyDict[Op, Op] = any_dict()
        for idx, placeholder in enumerate(self._signature_inputs()):
            mapping[placeholder] = FuncInputOp(
                param=Param.build(f"t{idx}"),
                typ=torch.Tensor,
                domain=TensorDomain(),
            )

        return OpFunc(self.output.replace(mapping))

    def _signature_inputs(self) -> list[TensorOp]:
        "The placeholders reachable from `output`, in post order."

        free = {id(placeholder) for placeholder in self.input_index.values()}
        return [
            op
            for op in _unique_post_order(self.output)
            if isinstance(op, TensorOp) and id(op) in free
        ]

    def _op_of(self, tensor: torch.Tensor) -> Op:
        "The `Op` of `tensor`, a new placeholder if not produced."

        if tensor in self.output_index:
            return self.output_index[tensor]

        if tensor not in self.input_index:
            self.input_index[tensor] = TensorOp(tensor)

        return self.input_index[tensor]


# Torch function / dispatch tracers ====


@dcls.dataclass
class _TorchCallOpDag(OpHist, abc.ABC):
    "The base class for torch functions or dispatches."

    def run(self, thunk):
        result = thunk()

        # Only track the function if it is point of interest.
        if self._track_thunk(thunk.func):
            thunk_op = self.lift(CallOp, thunk.func, thunk.args, thunk.kwargs)
            self.append(thunk_op, result)

        return result

    @abc.abstractmethod
    def _track_thunk(self, func) -> bool:
        raise NotImplementedError


@typing.final
@dcls.dataclass
class TorchFuncOpDag(_TorchCallOpDag, TorchFuncMode):
    """
    The DAG for `__torch_function__` calls.
    """

    def _track_thunk(self, func) -> bool:
        return is_torch_function(func) or is_tensor_method(func)


@typing.final
@dcls.dataclass
class _AtenOpDag(_TorchCallOpDag, TorchDispMode):
    """
    The DAG for `__torch_dispatch__` calls.

    Will only track `aten` ops, others are discarded.
    """

    def _track_thunk(self, func) -> bool:
        return is_aten_op(func)


@ctxl.contextmanager
def fake_aten_op_dag() -> cabc.Generator[OpHist]:
    """
    Test the DAG made from aten overrides.
    """

    dag = _AtenOpDag()

    with fake_mode(), route_aten_mode.activate(), dag.activate():
        yield dag


# Module tracer ====


@ctxl.contextmanager
def track_module_ops() -> cabc.Generator[ModuleOpHist]:
    tracker = ModuleOpTracker()

    with tracker():
        yield tracker.hist


class _ModuleInput(typing.NamedTuple):
    module: nn.Module
    input: typing.Any


@dcls.dataclass
class ModuleOpHist(OpHist):
    """
    The history to track input and output of a `nn.Module` during running.
    """

    def module_forward_hook(
        self, module: nn.Module, input: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        thunk = self.lift(ModuleThunkOp, module, input, {})
        self.append(thunk, output)


@dcls.dataclass(frozen=True)
class ModuleOpTracker:
    stack: Stack[_ModuleInput] = dcls.field(default_factory=Stack)
    "The current stack of modules."

    hist: ModuleOpHist = dcls.field(default_factory=ModuleOpHist)
    "The history of modules execution (only leaves are tracked)."

    @ctxl.contextmanager
    def __call__(self):
        with ctxl.ExitStack() as c:
            c.enter_context(register_module_forward_hook(self._fwd_post))
            c.enter_context(register_module_forward_pre_hook(self._fwd_pre))

            yield self

    def _fwd_pre(self, module: nn.Module, input) -> None:
        "The forward pre-hook."
        self.stack.append(_ModuleInput(module, input))

    def _fwd_post(self, module: nn.Module, input, output) -> None:
        "The forward hook (this executes after)."

        self._fwd_post_hist(module, input, output)
        self._fwd_post_stack(module, input)

    def _fwd_post_hist(self, module: nn.Module, input, output):
        """
        Update the history. This executes "before" the stack is popped,
        which means that `module` does exist in `self.stack`.

        Only append "leaf" modules, which means modules without parents,
        to prevent double counting.
        """

        assert isinstance(input, tuple)
        assert self.stack.top().module is module

        if not _is_leaf_module(module):
            return

        call = self.hist.lift(ModuleThunkOp, module, input, {})

        self.hist.append(call, output)

    def _fwd_post_stack(self, module: nn.Module, input):
        module_input = self.stack.top()

        # Sanity check to ensure that the modules and inputs are the right ones.
        assert module is module_input.module
        assert input is module_input.input

        self.stack.pop()


def _is_leaf_module(module: nn.Module) -> bool:
    return not list(module.children())


# Helper functions for the graph ====


def _result_ops(thunk: CallOp, result: typing.Any) -> AnyDict[torch.Tensor, Op]:
    """
    The `Op` of each tensor in `result`.

    Nested tensors are extracted from `thunk` by their own `CallOp`.
    """

    if isinstance(result, torch.Tensor):
        return any_dict((result, thunk))

    return any_dict(
        *[
            (leaf, _getter(thunk, path))
            for path, leaf in tree_leaves_with_path(result)
            if isinstance(leaf, torch.Tensor)
        ]
    )


def _getter(op: Op, path: KeyPath) -> CallOp:
    "The `CallOp` extracting the element at `path` of `op`'s result."

    if len(path) == 1:
        [key] = path

        if isinstance(key, SequenceKey):
            return CallOp(func=operator.getitem, args=(op, key.idx), kwargs={})

        if isinstance(key, MappingKey):
            return CallOp(func=operator.getitem, args=(op, key.key), kwargs={})

        if isinstance(key, GetAttrKey):
            return CallOp(func=getattr, args=(op, key.name), kwargs={})

    return CallOp(func=key_get, args=(op, path), kwargs={})


def _unique_post_order(op: Op) -> cabc.Iterator[Op]:
    """
    `Op.post_order`, but each `Op` is yielded once.

    First occurrences are in the same order as `Op.post_order`,
    without revisiting shared sub graphs (traced graphs share a lot).
    """

    seen: set[int] = set()

    def visit(node: Op) -> cabc.Iterator[Op]:
        if id(node) in seen:
            return

        seen.add(id(node))

        for dep in node.__deps__:
            yield from visit(dep)

        yield node

    yield from visit(op)
