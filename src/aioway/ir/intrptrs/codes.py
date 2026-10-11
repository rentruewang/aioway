# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import jinja2 as j2

from aioway.ir.instrs import FuncCall, InstrList
from aioway.ir.progs import FuncProgram
from aioway.t import TList

from .intrptrs import Intrptr

__all__ = ["CodeRender", "FuncProgramCode"]

_FUNCTION_TEMPLATE = """\
def {{ name }}({{ params }}):
{% for line in body %}
    {{ line }}
{% endfor %}
    return {{ returns }}
"""

_PY_ENV = j2.Environment(
    autoescape=False,  # Python source, not HTML.
    undefined=j2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


class CodeRender(typing.Protocol):
    def __str__(self) -> str: ...

    def __rich__(self) -> typing.Any: ...


# class _FuncRender:
#     def __rich__(self):
#         return syntax.Syntax(str(self), lexer="py")


@typing.final
class FuncProgramCode(Intrptr[FuncCall, str, CodeRender]):
    def __init__(self, program: FuncProgram, name: str) -> None:
        self._program = program
        self._name = name

    @typing.override
    def bind(self, inputs: TList, *args: str, **kwargs: str) -> None:
        raise NotImplementedError

    @typing.override
    def walk(self, instrs: InstrList[FuncCall]) -> None:
        raise NotImplementedError

    @typing.override
    def finalize(self) -> CodeRender:
        raise NotImplementedError

    @property
    @typing.override
    def program(self) -> FuncProgram:
        return self._program
