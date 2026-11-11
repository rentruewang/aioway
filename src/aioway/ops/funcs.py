# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import inspect
import typing

import tensordict as td
import torch
from torch import nn

from aioway._signs import BoundArgs, Param, Sign
from aioway._utils import AnyDict, any_dict

from .execs import OpExec, executor, set_executor
from .ops import Op
from .thunks import ModuleOp
from .vars import ValueOp

__all__ = ["OpFunc", "OpFuncEnv", "FuncInputOp"]


class OpFunc(nn.Module):
    """
    The function kind for `Op`.

    It traces from an output node until `FuncInputOp`s are found,
    which is treated as part of the signature.

    The signature would then be defined by the `FuncInputOp`
    that are found in post order traversal.

    On `__call__`, that signature is bound to the `__call__(*args, **kwargs)`
    to produce the input, which is then fed to `OpFuncEnv` to execute.
    """

    def __init__(self, output: Op) -> None:
        super().__init__()

        self._output = output

        # Register the sub modules.
        modules = [op.nn for op in self._all_ops() if isinstance(op, ModuleOp)]
        self.module_list = nn.ModuleList(modules)

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
        memo: AnyDict[typing.Any, object] = any_dict()
        for node in self._input_ops():
            memo[node] = bound[node.name]

        new = memo.replace(self.output)
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
        from .codes import FuncCodePrint

        return FuncCodePrint(self)()

    def _input_ops(self):
        # The dictionary is ordered.
        args: dict[str, FuncInputOp] = {}

        for op in self._all_ops():
            if not isinstance(op, FuncInputOp):
                continue

            if op.name in args:
                if args[op.name] != op:
                    raise TypeError("Different parameters defined with the same name.")

                continue

            args[op.name] = op
        return args.values()

    @property
    def signature(self) -> Sign:
        return Sign.from_params([i.param for i in self._input_ops()])

    @property
    def output(self) -> Op:
        return self._output

    def _all_ops(self):
        yield from self.output.post_order()


@dcls.dataclass(frozen=True)
class OpFuncEnv(OpExec):
    """
    The environment mapping that holds the function's input.
    """

    bounded: BoundArgs
    "The arguments that are bounded."

    fallback: OpExec = dcls.field(default_factory=lambda: executor().exec)
    """
    The fallback executor to use.
    By default, use the "current default" at the time of init.
    """

    def __call__(self, op: Op) -> typing.Any:
        input_mapping: AnyDict[FuncInputOp, ValueOp] = any_dict()
        for sub_op in op.post_order():
            if isinstance(sub_op, FuncInputOp):
                input_mapping[sub_op] = ValueOp(self.bounded[sub_op.name])

        # Replace placeholder with concrete values.
        replaced = input_mapping.replace(op)

        with set_executor(self.fallback):
            return replaced()

    def reset(self) -> None:
        # Results depend on `bounded`, so they must not outlive this call.
        self.fallback.reset()

    def __len__(self) -> int:
        return len(self.bounded)

    def __getitem__(self, key: str) -> typing.Any:
        return self.bounded[key]

    def __iter__(self):
        return iter(self.bounded.arguments)


class FuncInputOp[T](Op[T]):
    param: Param
    "The parameter that is associated with the input."

    typ: type[T]
    "The type that is accepted."

    @typing.override
    def execute(self) -> T:
        raise RuntimeError(
            "`FuncInputOp` is a placeholder, and never supposed to be called."
        )

    @typing.override
    def deps(self):
        return ()

    @property
    def name(self) -> str:
        return self.param.name
