# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import functools
import itertools
import typing
from collections import abc as cabc

import jinja2 as j2
import loguru as L
import torch
from rich import syntax

from aioway._utils import AnyDict, any_dict, render_fcall
from aioway.ir.instrs import FuncCall
from aioway.ir.instrs.instrs import InstrList
from aioway.ir.procs import Proc
from aioway.t import (
    Sign,
    keystr,
    render_torch_func_name,
    tree_flatten_with_path,
    tree_map_only,
)
from aioway.t.signs import BoundArgs

__all__ = ["CodeRender", "render_func_proc"]


class CodeRender(typing.Protocol):
    def __str__(self) -> str: ...

    def __rich__(self) -> typing.Any: ...


def render_func_proc(proc: Proc, name: str = "") -> CodeRender:
    ctx = proc.rename(name) if name else ctxl.nullcontext()

    with ctx:
        return FuncCodePrint(proc=proc, vars=VarName())()


# The renderer itself ====


class VarName:
    """
    The convenient variable name manager.
    """

    def __init__(
        self,
        vars: AnyDict[typing.Any, str] | None = None,
        count: cabc.Iterator | None = None,
    ) -> None:
        self._vars: AnyDict[typing.Any, str] = vars or any_dict()
        self._count: cabc.Iterator[int] = itertools.count(1) if count is None else count

    def __contains__(self, obj) -> bool:
        return obj in self._vars

    def __repr__(self) -> str:
        return repr(self._vars)

    def __len__(self) -> int:
        return len(self._vars)

    def __getitem__(self, var: typing.Any) -> str:
        return self._vars[var]

    def __setitem__(self, var: typing.Any, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Expected a string, but {name=} is  not a string.")

        self._vars[var] = name
        self.index_sub_attr(var)

    def index_sub_attr(self, tree: typing.Any) -> None:
        "Name the tensors inside `var` by their path from it, e.g. `tdict['a']`."

        if tree not in self:
            raise KeyError(f"{tree=} is not named yet.")

        for path, leaf in tree_flatten_with_path(tree)[0]:
            # A tensor that already has a name keeps it.
            if isinstance(leaf, torch.Tensor) and leaf not in self:
                self._vars[leaf] = self[tree] + keystr(path)

    def get_or_name(self, var: typing.Any) -> str:
        """
        Get the name if found. Or else create a new name (with the `new` function).
        """

        if var not in self:
            # Using `__setitem__` s.t. sub tensors are also named.
            self[var] = self.new(var)

        return self[var]

    def new(self, var) -> str:
        """
        Create a new name for the variable.

        `tensor_<n>` for tensors, the lowercase type name for others.
        """
        typ = torch.Tensor if isinstance(var, torch.Tensor) else type(var)
        prefix = typ.__name__.lower()
        next_int = next(self._count)
        return f"{prefix}_{next_int}"


class StmtCodePrint:
    """
    Render multiple statements.
    """

    def __init__(self, instrs: InstrList, vars: VarName) -> None:
        self._instrs = instrs
        self._steps: list[str] = []
        self._vars = vars


class FuncCodePrint:
    """
    Render the function as a python function.

    Tensors inside other objects (inputs or results) are named by their path,
    e.g. `tdict['a']`.
    """

    TYPE = str

    def __init__(self, proc: Proc, vars: VarName) -> None:
        self._proc = proc
        self._steps: list[str] = []
        self._vars = vars

    @typing.final
    def __call__(self):
        # Assign bound argument's arguments to its name in signature.
        for name, val in self.inputs.items():
            self._vars[val] = name

        L.logger.debug("Prepare walking the {} instructions.", len(self.instrs))

        # Execute the steps one by one in topo sorted order.
        for idx, instr in enumerate(self.instrs):
            self.step(idx, instr)

        L.logger.debug("Preparing the output.")

        return _FuncRender(
            name=self.proc.__name__,
            params=self._renamed_signature(),
            lines=self._steps,
            returns=self._render_return_obj_init(),
        )

    def step(self, idx: int, instr: FuncCall) -> None:
        # Rerun is not permitted.
        assert len(self._steps) == idx, "Length is not equal"

        args = tree_map_only(torch.Tensor, self._vars.__getitem__, instr.args)
        kwargs = tree_map_only(torch.Tensor, self._vars.__getitem__, instr.kwargs)

        lhs = self._vars.get_or_name(instr.result)

        func_name = render_torch_func_name(instr.func)
        self._steps.append(lhs + " = " + render_fcall(func_name, *args, **kwargs))

    @property
    def proc(self) -> Proc:
        return self._proc

    @property
    def instrs(self) -> InstrList:
        return self.proc.instrs

    @property
    def inputs(self) -> BoundArgs:
        return self.proc.inputs

    @property
    def outputs(self):
        return self.proc.outputs

    def _render_return_obj_init(self) -> str:
        # No pytree decomposition because some values we want may be decomposed.
        # E.g. mapping will not show up if we use pytree.
        def render(obj) -> str:
            # If the object is named, stop decomposing.
            if obj in self._vars:
                return self._vars[obj]

            # Tensors must be input or indexed already.
            if isinstance(obj, torch.Tensor):
                return self._vars[obj]

            if isinstance(obj, cabc.Mapping):
                builder = ["{"]
                for key, val in obj.items():
                    render_val = render(val)
                    builder.append(f"{key!r}:{render_val}")
                    builder.append(",")

                # Drop the `,`, but be careful about empty dicts.
                if len(builder) > 1:
                    assert builder[-1] == ","
                    builder.pop()
                builder.append("}")
                return "".join(builder)

            if isinstance(obj, cabc.Set):
                body = ", ".join(render(elem) for elem in obj)
                return "{" + body + "}"

            if isinstance(obj, cabc.Sequence):
                body = ", ".join(render(elem) for elem in obj)
                return "[" + body + "]"

            if dcls.is_dataclass(obj):
                obj_type = type(obj).__name__
                fields = dcls.fields(obj)
                body = ",".join(f"{f.name}={getattr(obj,f.name)}" for f in fields)
                return obj_type + "(" + body + ")"

            # Default, I give up trying.
            return repr(obj)

        return render(self.outputs)

    def _renamed_signature(self) -> Sign:
        # Get the original inputs (with symbolic data), and find out their new names.
        rename_map = {key: self._vars[val] for key, val in self.inputs.items()}
        return self.inputs.sign.rename(**rename_map)


# Convenient object that can be used in different contexts ====


@dcls.dataclass(frozen=True, slots=False)
class _FuncRender:
    name: str
    params: Sign
    lines: list[str]
    returns: str

    def __str__(self) -> str:
        return _function_renderer()(
            name=self.name,
            params=str(self.params),
            body=self.lines,
            returns=self.returns,
        )

    def __rich__(self):
        return syntax.Syntax(str(self), lexer="py")


def _function_template():
    return """\
def {{ name }}{{ params }}:
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
