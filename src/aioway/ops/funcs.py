# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import inspect
import typing

import tensordict as td
import torch
from torch import nn

from aioway._signs import BoundArgs, Param, Sign
from aioway._utils import AnyDict, any_dict
from aioway.t import AnyDomain, Domain

from .execs import OpExec, executor, set_executor
from .ops import Op
from .thunks import ModuleOp
from .vars import ValueOp
lazy from .codes import render_op_module

__all__ = ["OpFunc", "OpFuncModule", "OpFuncEnv", "FuncInputOp"]


class OpFunc:
    """
    The function kind for `Op`.

    It traces from an output node until `FuncInputOp`s are found,
    which is treated as part of the signature.

    The signature would then be defined by the `FuncInputOp`
    that are found in post order traversal.

    On `__call__`, that signature is bound to the `__call__(*args, **kwargs)`
    to produce the input, which is then fed to `OpFuncEnv` to execute.

    Use `module` for the `nn.Module` view (parameters, `state_dict`, etc.).
    """

    def __init__(self, output: Op) -> None:
        self._output = output

        # Compute signature, may also raise errors.
        _ = self.signature

    def __call__(self, *args, **kwargs) -> typing.Any:
        bound = self.signature.bind(*args, **kwargs)
        env = OpFuncEnv(bound)

        with set_executor(env):
            return self.output()

    def apply(self, *args, **kwargs) -> typing.Self:
        """
        Apply the function onto some `Op`s, merging into a bigger expression tree.

        Accepts the same signature as the `__call__` method,
        with `Domain` types replaced by `Op`s.
        """

        # FIXME: Op type as lazy Domain #737
        bound = self.signature.bind(*args, **kwargs)

        for arg in bound.values():
            if isinstance(arg, torch.Tensor) or td.is_tensor_collection(arg):
                raise TypeError("The tensor objects should be wrapped.")

        # The memo for replacement.
        memo: AnyDict[Op, Op] = any_dict()
        for node in self._input_ops():
            memo[node] = ValueOp(bound[node.name])

        new = self.output.replace(memo)
        return type(self)(new)

    def __str__(self) -> str:
        return str(self._codes)

    def __rich__(self):
        return self._codes

    @property
    def __signature__(self) -> inspect.Signature:
        return self.signature.to_inspect()

    @property
    def _codes(self):
        return render_op_module(self)

    def _input_ops(self) -> list[FuncInputOp]:
        """
        All the `FuncInputOp`s, including equal copies sharing the same name.

        Raises:
            TypeError: if different parameters share the same name.
        """

        # Return all the input nodes. Since different nodes can have the same param,
        # ensure all input nodes with the same name are the same param,
        # so check against the first occurence of the name.
        first: dict[str, FuncInputOp] = {}
        inputs: list[FuncInputOp] = []

        for op in self._all_ops():
            if not isinstance(op, FuncInputOp):
                continue

            if op.name in first and first[op.name] != op:
                raise TypeError("Different parameters defined with the same name.")

            first.setdefault(op.name, op)
            inputs.append(op)

        return inputs

    @property
    def signature(self) -> Sign:
        # One parameter per name, ordered by first appearance.
        params = {op.name: op.param for op in self._input_ops()}
        return Sign.from_params(params.values())

    @property
    def output(self) -> Op[typing.Any]:
        return self._output

    @property
    def module(self) -> "OpFuncModule":
        "The `nn.Module` view, built from the current graph."
        return OpFuncModule(self)

    def _all_ops(self):
        yield from self.output.post_order()


class OpFuncModule(nn.Module):
    """
    The `nn.Module` view of `OpFunc`, for parameter visibility.

    Cheap to construct, registers the `nn.Module`s in the graph at construction.
    """

    def __init__(self, func: OpFunc) -> None:
        super().__init__()

        self.func = func
        self.module_list = nn.ModuleList(
            op.net for op in func._all_ops() if isinstance(op, ModuleOp)
        )

    def forward(self, *args, **kwargs) -> typing.Any:
        return self.func(*args, **kwargs)


@dcls.dataclass(frozen=True)
class OpFuncEnv(OpExec):
    """
    The environment mapping that holds the function's input.
    """

    bounded: BoundArgs
    "The arguments that are bounded."

    fallback: OpExec = dcls.field(default_factory=executor)
    """
    The fallback executor to use.
    By default, use the "current default" at the time of init.
    """

    def __call__(self, op: Op) -> typing.Any:
        # Mapping from `FuncInputOp` to a `ValueOp`.
        input_mapping: AnyDict[Op, Op] = any_dict()
        for sub_op in op.post_order():
            if not isinstance(sub_op, FuncInputOp):
                continue

            # Perform checks here.
            sub_op.domain.check(value := self.bounded[sub_op.name])
            input_mapping[sub_op] = ValueOp(value)

        # Replace placeholder with concrete values.
        replaced = op.replace(input_mapping)

        with set_executor(self.fallback):
            return replaced()

    def __len__(self) -> int:
        return len(self.bounded)

    def __getitem__(self, key: str) -> typing.Any:
        return self.bounded[key]

    def __iter__(self):
        return iter(self.bounded.arguments)


class FuncInputOp[T](Op[T]):
    param: Param
    "The parameter that is associated with the input."

    domain: Domain = dcls.field(default_factory=AnyDomain)
    "The domain field."

    @property
    @typing.override
    def __domain_deps__(self) -> list[Domain]:
        return []

    @typing.override
    def _infer_domain(self) -> Domain:
        return self.domain

    @property
    @typing.override
    def __deps__(self) -> tuple[()]:
        return ()

    @typing.override
    def execute(self) -> T:
        raise RuntimeError(
            "`FuncInputOp` is a placeholder, and never supposed to be called."
        )

    @property
    def name(self) -> str:
        return self.param.name
