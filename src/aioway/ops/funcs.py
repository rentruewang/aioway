# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import inspect
import typing

from torch import nn

from aioway._signs import BoundArgs, Param, Sign

from .execs import CachedOpExec, OpExec, set_executor
from .ops import Op, op_dcls
from .thunks import ModuleOp

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
        env = OpFuncEnv(self.signature.bind(*args, **kwargs))

        with set_executor(env):
            return self.output()

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

    @functools.cached_property
    def signature(self) -> Sign:
        def is_input(op) -> typing.TypeIs[FuncInputOp]:
            return isinstance(op, FuncInputOp)

        input_set: set[FuncInputOp] = set()
        input_list: list[FuncInputOp] = []

        for op in self._all_ops():
            if not is_input(op):
                continue

            if op in input_set:
                continue

            input_set.add(op)
            input_list.append(op)

        return Sign.from_params([i.param for i in input_list])

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

    fallback: OpExec = dcls.field(default_factory=CachedOpExec)
    """
    The fallback executor to use.
    By default, use the "current default" at the time of init. 
    """

    def __call__(self, op: Op) -> typing.Any:
        if isinstance(op, FuncInputOp):
            return self.bounded[op.name]
        else:
            return self.fallback(op)

    def __len__(self) -> int:
        return len(self.bounded)

    def __getitem__(self, key: str) -> typing.Any:
        return self.bounded[key]

    def __iter__(self):
        return iter(self.bounded.arguments)


@op_dcls
class FuncInputOp[T](Op[T]):
    param: Param
    "The parameter that is associated with the input."

    typ: type[T]
    "The type that is accepted."

    @typing.override
    def execute(self) -> T:
        raise RuntimeError(
            "This function should enver be called, "
            "as `FuncInputOp` can only be executed by `OpFuncEnv`."
        )

    @typing.override
    def deps(self):
        return ()

    @property
    def name(self) -> str:
        return self.param.name
