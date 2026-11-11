# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import itertools
import textwrap
import typing
from collections import abc as cabc

import jinja2 as j2
import loguru as L
import torch
from rich import syntax

from aioway._pytrees import keystr, tree_flatten_with_path, tree_map_only
from aioway._signs import Sign
from aioway._utils import AnyDict, any_dict, render_fcall
from aioway.t import render_torch_func_name

from .funcs import FuncInputOp, OpFunc
from .ops import Op
from .thunks import CallOp, ModuleThunkOp

__all__ = ["CodeRender", "render_op_func", "render_op_module", "VarName"]


class CodeRender(typing.Protocol):
    "Rendered code, printable as text or with `rich`."

    def __str__(self) -> str:
        "The source code."

    def __rich__(self) -> typing.Any:
        "The source code, highlighted for `rich`."


def render_op_func(func: OpFunc, name: str = "forward") -> CodeRender:
    "Render the `OpFunc` as a python function."
    stmt_print = CallStmtPrint(vars=VarName())
    return _FunctionRenderer(func=func, stmt_print=stmt_print, name=name)()


def render_op_module(func: OpFunc, name: str = "Module") -> CodeRender:
    "Render the `OpFunc` as an `nn.Module` class."
    vars = VarName()
    modules = VarName()

    # Modules first, they are checked in order.
    stmt_print = StmtSwitch(vars=vars)
    stmt_print.register(ModuleStmtPrint(vars=vars, modules=modules))
    stmt_print.register(CallStmtPrint(vars=vars))

    forward = _FunctionRenderer(func=func, stmt_print=stmt_print)()
    return _ModuleRenderer(
        name=name, modules=list(modules._vars.values()), forward=forward
    )


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
        "Start from `vars` and `count` if given, or else empty."
        self._vars: AnyDict[typing.Any, str] = vars or any_dict()
        self._count: cabc.Iterator[int] = itertools.count(1) if count is None else count

    def __contains__(self, obj) -> bool:
        "Whether `obj` is named."
        return obj in self._vars

    def __repr__(self) -> str:
        "The names, as a dict."
        return repr(self._vars)

    def __len__(self) -> int:
        "The number of named objects."
        return len(self._vars)

    def __getitem__(self, var: typing.Any) -> str:
        "The name of `var`."
        return self._vars[var]

    def __setitem__(self, var: typing.Any, name: str) -> None:
        "Name `var`, and the tensors inside it."
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


# Print line ====


@dcls.dataclass(frozen=True)
class StmtCodePrint[T: Op = typing.Any](abc.ABC):
    """
    Render an op as a statement. One implementation per kind of op.
    """

    vars: VarName
    "The (local) variable manager."

    @abc.abstractmethod
    def __call__(self, op: T) -> str:
        "Convert an op to a statement."

    @property
    @abc.abstractmethod
    def op_type(self) -> type[T]:
        "The kind of op this renders."


class _FcallMixin(abc.ABC):
    """
    Mixin rendering an op as `lhs = func(...)`, for ops that are a single call.
    """

    vars: VarName

    def __call__(self, op: CallOp) -> str:
        "Convert an op to a `lhs = func(...)` statement."
        # First, s.t. an op of the wrong kind fails here.
        func_name = self.func_name(op)

        args = tree_map_only(Op, self.vars.__getitem__, op.args)
        kwargs = tree_map_only(Op, self.vars.__getitem__, op.kwargs)

        lhs = self.vars.get_or_name(op)

        return lhs + " = " + render_fcall(func_name, *args, **kwargs)

    @abc.abstractmethod
    def func_name(self, op: CallOp) -> str:
        "The name of the function the op calls."

        raise NotImplementedError


@dcls.dataclass(frozen=True)
class CallStmtPrint(_FcallMixin, StmtCodePrint[CallOp]):
    """
    Render a `CallOp` as a `func(...)` statement.
    """

    @property
    def op_type(self) -> type[CallOp]:
        "Renders `CallOp`."
        return CallOp

    def func_name(self, op: CallOp) -> str:
        "The torch function's name."
        assert isinstance(op, CallOp), f"Cannot render {type(op)=}."
        return render_torch_func_name(op.func)


@dcls.dataclass(frozen=True)
class ModuleStmtPrint(_FcallMixin, StmtCodePrint[ModuleThunkOp]):
    """
    Render a `ModuleThunkOp` as a `self.<module>(...)` statement.
    """

    modules: VarName
    "The `self.<name>` manager, for modules."

    @property
    def op_type(self) -> type[ModuleThunkOp]:
        "Renders `ModuleThunkOp`."
        return ModuleThunkOp

    def func_name(self, op: CallOp) -> str:
        "The module's name, `self.<module>`."
        assert isinstance(op, ModuleThunkOp), f"Cannot render {type(op)=}."
        return "self." + self.modules.get_or_name(op.nn)


@dcls.dataclass(frozen=True)
class StmtSwitch(StmtCodePrint[Op]):
    """
    Render each op with the printer registered for its type.
    """

    cases: list[StmtCodePrint] = dcls.field(default_factory=list)
    "The printers, checked in the order they are registered."

    @property
    def op_type(self) -> type[Op]:
        "Renders any op, given a registered printer for it."
        return Op

    def register(self, case: StmtCodePrint) -> None:
        "Render ops of `case.op_type` with `case`, unless an earlier case matches."
        self.cases.append(case)

    def __call__(self, op: Op) -> str:
        "Render with the first printer whose `op_type` matches."
        for case in self.cases:
            if isinstance(op, case.op_type):
                return case(op)

        raise TypeError(f"Cannot render {type(op)=}.")


class _FunctionRenderer:
    """
    Render the `OpFunc` as a python function.

    Tensors inside other objects (inputs or results) are named by their path,
    e.g. `tdict['a']`.
    """

    TYPE = str
    "The type of the rendered code."

    def __init__(
        self,
        func: OpFunc,
        stmt_print: StmtCodePrint,
        name: str = "forward",
    ) -> None:
        "Render `func`, with `stmt_print` for each statement."
        self._func = func
        self._stmt_print = stmt_print
        self._name = name

    @typing.final
    def __call__(self):
        "Render the function."
        # Name each input after its parameter in the signature.
        for op in self.inputs:
            self.vars[op] = op.param.name

        L.logger.debug("Prepare walking the {} instructions.", len(self.instrs))

        # Render the steps one by one in topo sorted order.
        body_lines = [self._stmt_print(op) for op in self.instrs]

        L.logger.debug("Preparing the output.")

        return _FuncRender(
            name=self._name,
            params=self.func.signature,
            lines=body_lines,
            returns=self._render_return_obj_init(),
        )

    @property
    def vars(self) -> VarName:
        "The variable manager, from the strategy."
        return self._stmt_print.vars

    @property
    def func(self) -> OpFunc:
        "The function to render."
        return self._func

    @property
    def ops(self) -> list[Op]:
        "All ops, linearly ordered (post order)."
        return list(self.func.output.post_order())

    @property
    def instrs(self) -> list[Op]:
        "The ops rendered as statements, i.e. all but the inputs."
        return [op for op in self.ops if not isinstance(op, FuncInputOp)]

    @property
    def inputs(self) -> list[FuncInputOp]:
        "The function's inputs."
        return [op for op in self.ops if isinstance(op, FuncInputOp)]

    @property
    def outputs(self):
        "The function's output."
        return self.func.output

    def _render_return_obj_init(self) -> str:
        "Render the output as the `return` expression."

        # No pytree decomposition because some values we want may be decomposed.
        # E.g. mapping will not show up if we use pytree.
        def render(obj) -> str:
            "Render `obj`, decomposing containers until a named object."
            # If the object is named, stop decomposing.
            if obj in self.vars:
                return self.vars[obj]

            # Tensors must be input or indexed already.
            if isinstance(obj, torch.Tensor):
                return self.vars[obj]

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


# Convenient object that can be used in different contexts ====


@dcls.dataclass(frozen=True, slots=False)
class _FuncRender:
    "A rendered python function."

    name: str
    "The function name."

    params: Sign
    "The function signature."

    lines: list[str]
    "The body statements."

    returns: str
    "The `return` expression."

    def __str__(self) -> str:
        "The function's source code."
        return _function_renderer()(
            name=self.name,
            param_list=str(self.params),
            body=self.lines,
            returns=self.returns,
        )

    def __rich__(self):
        "The source code, highlighted for `rich`."
        return syntax.Syntax(str(self), lexer="py")


@dcls.dataclass(frozen=True, slots=False)
class _ModuleRenderer:
    "A rendered `nn.Module` class."

    name: str
    "The class name."

    modules: list[str]
    "The submodule names, set in `__init__`."

    forward: _FuncRender
    "The `forward` method, without `self`."

    def __str__(self) -> str:
        "The class's source code, with `self` added to `forward`."
        forward = _function_renderer()(
            name=self.forward.name,
            param_list=_with_self(str(self.forward.params)),
            body=self.forward.lines,
            returns=self.forward.returns,
        )
        return _module_renderer()(name=self.name, modules=self.modules, forward=forward)

    def __rich__(self):
        "The source code, highlighted for `rich`."
        return syntax.Syntax(str(self), lexer="py")


def _with_self(params: str) -> str:
    "`(x, y)` -> `(self, x, y)`, `()` -> `(self)`."
    if params.startswith("()"):
        return "(self" + params[1:]
    return "(self, " + params[1:]


@dcls.dataclass(frozen=True, slots=False)
class _StmtsRender:
    "Rendered python statements."

    body: list[str]
    "The statements."

    def __str__(self) -> str:
        "The statements' source code."
        return _stmts_renderer()(body=self.body)

    def __rich__(self):
        "The source code, highlighted for `rich`."
        return syntax.Syntax(str(self), lexer="py")


def _dedent(string: str):
    return textwrap.dedent(string.strip("\n"))


def _function_template():
    return _dedent("""
        def {{ name }}{{ param_list }}:
            {% for line in body %}
            {{ line }}
            {% endfor %}
            return {{ returns }}
""")


def _statements_template():
    return _dedent("""
        {% for line in body %}
        {{ line }}
        {% endfor %}
    """)


def _module_template():
    return _dedent("""
        class {{ name }}(nn.Module):
            def __init__(self{% for m in modules %}, {{ m }}{% endfor %}):
                super().__init__()
                {% for m in modules %}
                self.{{ m }} = {{ m }}
                {% endfor %}

        {{ forward | indent(4, first=True) }}
    """)


def _py_env():
    "The j2 environment for rendering python source."

    return j2.Environment(
        autoescape=False,  # Python source, not HTML.
        undefined=j2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


@functools.cache
def _stmts_renderer():
    return _py_env().from_string(_statements_template()).render


@functools.cache
def _function_renderer():
    return _py_env().from_string(_function_template()).render


@functools.cache
def _module_renderer():
    return _py_env().from_string(_module_template()).render
