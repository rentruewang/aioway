# Copyright (c) AIoWay Authors - All Rights Reserved

"Render a `Program` as the source of a Python function, with Jinja2 templates."

import dataclasses as dcls
import enum
import keyword
import re
import types
import typing
from collections import abc as cabc

import jinja2
import torch

from aioway._utils import AnyDict, any_dict
from aioway.ir.instrs import Instr
from aioway.t import parse_attr

__all__ = [
    "ProgramLike",
    "Fmt",
    "ExprFn",
    "FUNCTION_TEMPLATE",
    "CALL_TEMPLATE",
    "Renderer",
    "render",
]


class ProgramLike(typing.Protocol):
    """
    What the renderer needs from a `Program`.

    Typed structurally so this module never imports the program module,
    which leaves `Program` free to import the renderer (e.g. for `__str__`).
    """

    @property
    def instrs(self) -> cabc.Iterable[Instr]: ...

    @property
    def inputs(self) -> cabc.Iterable[torch.Tensor]: ...

    @property
    def outputs(self) -> cabc.Iterable[torch.Tensor]: ...


type Fmt = cabc.Callable[[typing.Any], str]
"Turns a value into source: tensors become variable names, the rest become literals."

type ExprFn[I: Instr] = cabc.Callable[[I, Fmt], str]
"A custom right-hand side for one instruction type, see `Renderer.register`."


# The templates ====


FUNCTION_TEMPLATE = """\
def {{ name }}({{ params }}):
{% for line in body %}
    {{ line }}
{% endfor %}
    return {{ returns }}
"""
"""
The whole function. Context:

- `name`: The function name.
- `params`: The parameters, e.g. `"t0, t1"`.
- `body`: The lines of the body, unindented.
- `returns`: The returned tuple, e.g. `"(t3,)"`.
"""

CALL_TEMPLATE = "{{ op }}({{ args }})"
"""
The default right-hand side of an instruction. Context:

- `op`: The class name in snake case, trailing `Instr` dropped.
- `args`: The arguments, e.g. `"t0, dim=1"`.
"""


# The renderer ====


class Renderer:
    """
    Renders a `Program` as a Python function.

    The program inputs are the parameters, every instruction is one statement
    binding its outputs, and the program outputs are returned as a tuple::

        def program(t0, t1):
            # t0: <parse_attr(t0)>
            # t1: <parse_attr(t1)>
            t2 = mat_mul(lhs=t0, rhs=t1)  # <parse_attr(t2)>
            t3 = relu(x=t2)  # <parse_attr(t3)>
            return (t3,)

    By default an instruction renders as a call named after its class in snake case,
    with one keyword argument per dataclass field.
    Tensors inside fields, including inside tuples, lists, dicts, namedtuples and
    dataclasses, render as variable names.
    Fields holding only the instruction's own outputs are left out,
    since those are already the left-hand side.
    Inputs that no field holds are passed positionally, so no dependency is hidden.

    For a bare instruction list, render `Program.from_instr_list(instrs)`.
    """

    def __init__(
        self,
        *,
        prefix: str = "t",
        annotate: bool = True,
        template: str = FUNCTION_TEMPLATE,
        call: str = CALL_TEMPLATE,
    ) -> None:
        """
        Args:
            prefix: Variables are named `prefix` followed by a counter.
            annotate: Comment every variable with its `parse_attr`.
            template: The Jinja2 template of the whole function.
            call: The Jinja2 template of an instruction without a registered one.
        """

        if not prefix.isidentifier() or keyword.iskeyword(prefix):
            raise ValueError(f"{prefix=} is not a valid identifier.")

        self._prefix = prefix
        self._annotate = annotate
        self._env = _environment()
        self._function = self._env.from_string(template)
        self._call_template = self._env.from_string(call)
        self._exprs: dict[type, ExprFn] = {}

    def register[I: Instr](self, cls: type[I], expr: str | ExprFn[I], /) -> None:
        """
        Override the right-hand side for `cls` and its subclasses.

        Args:
            cls: The instruction type.
            expr: A Jinja2 template whose variables are the instruction's fields,
                already rendered as source (tensors as variable names),
                e.g. `"{{ lhs }} @ {{ rhs }}"`.
                Or a callable `(instr, fmt) -> str`.
        """

        if isinstance(expr, str):
            template = self._env.from_string(expr)

            def from_template(instr: I, fmt: Fmt) -> str:
                return template.render(_fields(instr, fmt)).strip()

            expr = from_template

        self._exprs[cls] = expr

    def render(self, program: ProgramLike, /, *, name: str = "program") -> str:
        """
        Render `program` as the source of a function.

        Args:
            program: The program, or anything with `instrs`, `inputs` and `outputs`.
            name: The name of the function.

        Raises:
            ValueError: If a step reads a tensor before it is defined,
                a tensor is defined twice, or an output is never defined.
        """

        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"{name=} is not a valid function name.")

        scope = _Scope(self._prefix)

        inputs = list(program.inputs)
        params = [scope.define(t) for t in inputs]

        body: list[str] = []

        if self._annotate:
            body += [f"# {p}: {parse_attr(t)!s}" for p, t in zip(params, inputs)]

        body += [self._statement(i, s, scope) for i, s in enumerate(program.instrs)]

        return self._function.render(
            name=name,
            params=", ".join(params),
            body=body,
            returns=_tuple(self._returns(list(program.outputs), scope)),
        )

    def _statement(self, step: int, instr: Instr, scope: "_Scope") -> str:
        for tensor in instr.inputs:
            if tensor not in scope:
                raise ValueError(
                    f"Step {step} ({type(instr).__name__}) reads a tensor that is "
                    "neither a program input nor the output of an earlier step."
                )

        outputs = list(instr.outputs)
        targets = [scope.define(t) for t in outputs]
        rhs = self._rhs(instr, fmt=scope.formatter(step, instr))

        line = f"{', '.join(targets)} = {rhs}" if targets else rhs

        if self._annotate and outputs:
            line += f"  # {', '.join(str(parse_attr(t)) for t in outputs)}"

        return line

    def _returns(self, outputs: list[torch.Tensor], scope: "_Scope") -> list[str]:
        for tensor in outputs:
            if tensor not in scope:
                raise ValueError("A program output is never defined.")

        return [scope[t] for t in outputs]

    def _rhs(self, instr: Instr, fmt: Fmt) -> str:
        # Walk the MRO so a registration covers subclasses too.
        for cls in type(instr).__mro__:
            if (fn := self._exprs.get(cls)) is not None:
                return fn(instr, fmt)

        return self._call(instr, fmt)

    def _call(self, instr: Instr, fmt: Fmt) -> str:
        "The default right-hand side: `snake_name(field=value, ...)`."

        outputs = list(instr.outputs)
        held: list[torch.Tensor] = []
        kwargs: list[str] = []

        for field in dcls.fields(instr):
            if not field.init:
                continue

            value = getattr(instr, field.name)
            tensors = _tensors(value)

            # Fields holding only outputs are the left-hand side already.
            if tensors and all(_has(outputs, t) for t in tensors):
                continue

            held += tensors
            kwargs.append(f"{field.name}={fmt(value)}")

        # Inputs no field holds (e.g. computed in `_inputs`) still have to show.
        positional = [fmt(t) for t in instr.inputs if not _has(held, t)]

        return self._call_template.render(
            op=self._op_name(type(instr)), args=", ".join([*positional, *kwargs])
        )

    def _op_name(self, cls: type) -> str:
        base = cls.__name__.removesuffix("Instr") or cls.__name__
        name = _SNAKE.sub("_", base).lower()

        # Must not shadow a variable, nor be a keyword.
        if keyword.iskeyword(name) or re.fullmatch(rf"{self._prefix}\d+", name):
            name += "_"

        return name


def render(
    program: ProgramLike, /, *, name: str = "program", annotate: bool = True
) -> str:
    "Render `program` as a Python function with the default `Renderer`."

    return Renderer(annotate=annotate).render(program, name=name)


# Helpers ====


_SNAKE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
"Word boundaries in CamelCase: `MatMul` -> `Mat_Mul`, `HTTPGet` -> `HTTP_Get`."


def _environment() -> jinja2.Environment:
    return jinja2.Environment(
        autoescape=False,  # Python source, not HTML.
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


class _Scope:
    "Variable names of the tensors defined so far, keyed by identity."

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._names: AnyDict[torch.Tensor, str] = any_dict(torch.Tensor)
        self._count = 0

    def __contains__(self, tensor: torch.Tensor) -> bool:
        return tensor in self._names

    def __getitem__(self, tensor: torch.Tensor) -> str:
        return self._names[tensor]

    def define(self, tensor: torch.Tensor) -> str:
        if tensor in self._names:
            raise ValueError(f"Tensor `{self._names[tensor]}` is defined twice.")

        name = f"{self._prefix}{self._count}"
        self._count += 1
        self._names[tensor] = name
        return name

    def formatter(self, step: int, instr: Instr) -> Fmt:
        "The `Fmt` for one step, whose errors say which step went wrong."

        def name_of(tensor: torch.Tensor) -> str:
            if tensor not in self._names:
                raise ValueError(
                    f"Step {step} ({type(instr).__name__}) holds a tensor that is "
                    "neither a program input nor the output of an earlier step."
                )

            return self._names[tensor]

        return lambda value: _source(value, name_of)


def _fields(instr: Instr, fmt: Fmt) -> dict[str, str]:
    "Each init field of `instr`, rendered as source."

    return {f.name: fmt(getattr(instr, f.name)) for f in dcls.fields(instr) if f.init}


def _tuple(items: list[str]) -> str:
    "Source of a tuple of already rendered items."

    return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"


def _source(value: typing.Any, name_of: cabc.Callable[[torch.Tensor], str]) -> str:
    "Python source for `value`, with each tensor replaced by `name_of(tensor)`."

    def src(v: typing.Any) -> str:
        return _source(v, name_of)

    match value:
        case torch.Tensor():
            return name_of(value)

        # Namedtuple, before plain tuple.
        case tuple() if hasattr(value, "_fields"):
            args = ", ".join(f"{k}={src(v)}" for k, v in zip(value._fields, value))
            return f"{type(value).__qualname__}({args})"

        case tuple() if type(value) is tuple:
            return _tuple([src(v) for v in value])

        case list() if type(value) is list:
            return f"[{', '.join(src(v) for v in value)}]"

        case dict() if type(value) is dict:
            return (
                "{" + ", ".join(f"{src(k)}: {src(v)}" for k, v in value.items()) + "}"
            )

        case _ if dcls.is_dataclass(value) and not isinstance(value, type):
            fields = (f for f in dcls.fields(value) if f.init)
            args = ", ".join(f"{f.name}={src(getattr(value, f.name))}" for f in fields)
            return f"{type(value).__qualname__}({args})"

        case enum.Enum():
            return f"{type(value).__qualname__}.{value.name}"

        case type() | types.FunctionType() | types.BuiltinFunctionType():
            return value.__qualname__

        case _:
            return repr(value)


def _tensors(value: typing.Any) -> list[torch.Tensor]:
    "The tensors in `value`, found by the same walk that `_source` does."

    found: list[torch.Tensor] = []

    def collect(tensor: torch.Tensor) -> str:
        found.append(tensor)
        return ""

    _source(value, collect)
    return found


def _has(tensors: cabc.Iterable[torch.Tensor], tensor: torch.Tensor) -> bool:
    "Identity membership, as `==` on tensors is elementwise."

    return any(t is tensor for t in tensors)
