# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import itertools
import typing

import jinja2 as j2
import torch
from rich import syntax
from torch.utils import _pytree as pyt

from aioway._utils import AnyDict, any_dict, render_fcall_str
from aioway.ir.instrs import FuncCall
from aioway.ir.progs import FuncProgram
from aioway.t import TList, render_torch_func_name

from .intrptrs import Intrptr

__all__ = ["CodeRender", "render_stateless_program"]


class CodeRender(typing.Protocol):
    def __str__(self) -> str: ...

    def __rich__(self) -> typing.Any: ...


def render_stateless_program(
    program: FuncProgram, name: str, *args: str, **kwargs: str
) -> CodeRender:
    interpret = _FuncProgramCode(program=program, name=name)
    return interpret(*args, **kwargs)


# The renderer itself ====


class _FuncProgramCode(Intrptr[FuncCall, str, CodeRender]):
    """
    Render the function as a python function.

    Right now only supports `torch.Tensor` as intermediate output.
    """

    TYPE = str

    def __init__(self, program: FuncProgram, name: str) -> None:
        self._program = program
        self._name = name
        self._vars: AnyDict[torch.Tensor, str] = any_dict(torch.Tensor)
        self._steps: list[str] = []
        self._counter = itertools.count(start=1)

    @typing.override
    def bind(self, inputs: TList, args: list[str]) -> None:
        for i, a in zip(inputs, args):
            self._vars[i] = a

    @typing.override
    def step(self, idx: int, instr: FuncCall) -> None:
        # Rerun is not permitted.
        assert len(self._steps) == idx

        args = pyt.tree_map_only(torch.Tensor, self._tensor_name, instr.args)
        kwargs = pyt.tree_map_only(torch.Tensor, self._tensor_name, instr.kwargs)

        if not isinstance(instr.result, torch.Tensor):
            raise NotImplementedError("Only supports tensor outputs for now.")

        result = self._tensor_name(instr.result)

        func_name = render_torch_func_name(instr.func)
        self._steps.append(
            result + " = " + render_fcall_str(func_name, *args, **kwargs)
        )

    @typing.override
    def finalize(self) -> CodeRender:
        assert len(self.program.outputs) == 1
        return _FuncRender(
            name=self._name,
            params=[self._tensor_name(var) for var in self.program.inputs],
            lines=self._steps,
            returns=self._tensor_name(self.program.outputs[0]),
        )

    @property
    @typing.override
    def program(self) -> FuncProgram:
        return self._program

    def _tensor_name(self, var: torch.Tensor) -> str:
        if var not in self._vars:
            next_int = next(self._counter)
            name = f"tensor_{next_int}"
            self._vars[var] = name

        return self._vars[var]


# Convenient object that can be used in different contexts ====


@dcls.dataclass(frozen=True, slots=False)
class _FuncRender:

    name: str
    params: list[str]
    lines: list[str]
    returns: str

    def __str__(self) -> str:
        return _function_renderer()(
            name=self.name,
            params=", ".join(self.params),
            body=self.lines,
            returns=self.returns,
        )

    def __rich__(self):
        return syntax.Syntax(str(self), lexer="py")


def _function_template():
    return """\
def {{ name }}({{ params }}):
{% for line in body %}
    {{ line }}
{% endfor %}
    return {{ returns }}
"""


def _py_env():
    return j2.Environment(
        autoescape=False,  # Python source, not HTML.
        undefined=j2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


@functools.cache
def _function_renderer():
    return _py_env().from_string(_function_template()).render
