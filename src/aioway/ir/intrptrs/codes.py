# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import itertools
import typing

import jinja2 as j2
import torch
from rich import syntax
from torch.utils import _pytree as pyt

from aioway._utils import AnyDict, any_dict, render_fcall
from aioway.ir.instrs import FuncCall
from aioway.ir.progs import FuncProgram, ProgramIn
from aioway.t import BoundArgs, render_torch_func_name

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

    Tensors inside other objects (inputs or results) are named by their path,
    e.g. `tdict['a']`.
    """

    TYPE = str

    def __init__(self, program: FuncProgram, name: str) -> None:
        self._program = program
        self._func_name = name
        self._vars: AnyDict[typing.Any, str] = any_dict()
        self._steps: list[str] = []
        self._counter = itertools.count(start=1)

    @typing.override
    def bind(self, inputs: ProgramIn, bound: BoundArgs) -> None:
        for i, a in zip(inputs, bound.values()):
            self._vars[i] = a
            self._decompose(i)

    @typing.override
    def step(self, idx: int, instr: FuncCall) -> None:
        # Rerun is not permitted.
        assert len(self._steps) == idx, "Length is not equal"

        args = pyt.tree_map_only(torch.Tensor, self.name, instr.args)
        kwargs = pyt.tree_map_only(torch.Tensor, self.name, instr.kwargs)

        self._vars[instr.result] = self._new_name(instr.result)
        self._decompose(instr.result)
        result = self.name(instr.result)

        func_name = render_torch_func_name(instr.func)
        self._steps.append(result + " = " + render_fcall(func_name, *args, **kwargs))

    @typing.override
    def finalize(self) -> CodeRender:
        assert len(self.program.outputs) == 1, "Length is not 1."
        return _FuncRender(
            name=self._func_name,
            params=[self.name(t) for t in self.program.inputs],
            lines=self._steps,
            returns=self.name(self.program.outputs[0]),
        )

    @property
    @typing.override
    def program(self) -> FuncProgram:
        return self._program

    def name(self, var: typing.Any) -> str:
        r = self._vars[var]
        assert isinstance(r, str)
        return r

    def _new_name(self, var: typing.Any) -> str:
        "`tensor_<n>` for tensors, the lowercase type name for others."

        typ = torch.Tensor if isinstance(var, torch.Tensor) else type(var)
        prefix = typ.__name__.lower()
        return f"{prefix}_{next(self._counter)}"

    def _decompose(self, var: typing.Any) -> None:
        "Name the tensors inside `var` by their path from it, e.g. `tdict['a']`."

        for path, leaf in pyt.tree_flatten_with_path(var)[0]:
            # A tensor that already has a name keeps it.
            if isinstance(leaf, torch.Tensor) and leaf not in self._vars:
                self._vars[leaf] = self.name(var) + pyt.keystr(path)


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
