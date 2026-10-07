# Copyright (c) AIoWay Authors - All Rights Reserved

"Render a `Program` as the source of a Python function, with Jinja2 templates."

import functools
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
from .progs import Program
from aioway.ir.instrs import Instr, InstrList, FuncCall, ModuleCall
import jinja2 as j2

__all__ = []

_ENV = j2.Environment(
    # Python source, not HTML.
    autoescape=False,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

_FUNCTION_TEMPLATE = """
def {{ name }}({{ params }}):
{% for line in body %}
    {{ line }}
{% endfor %}
    return {{ returns }}
""".lstrip()


def render_function(name: str, params: list[str], body: list[str], returns: str) -> str:
    return _function_template().render(
        name=name, params=",".join(params), body=body, returns=returns
    )


def render_program(program: Program) -> str:
    inputs = program.inputs
    instrs = program.instrs

    all_tensors = program.tensors


@functools.cache
def _function_template():
    return _ENV.from_string(_FUNCTION_TEMPLATE)
