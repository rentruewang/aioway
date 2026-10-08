# Copyright (c) AIoWay Authors - All Rights Reserved

"Render a `Program` as the source of a Python function, with Jinja2 templates."

import functools

import jinja2
import jinja2 as j2

from .progs import Program

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


def render_program(program: Program, name: str, args: list[str], var_names) -> str:
    inputs = program.inputs
    instrs = program.instrs

    if len(args) != len(inputs):
        raise ValueError

    all_tensors = program.tensors

    raise NotImplementedError


@functools.cache
def _function_template():
    return _ENV.from_string(_FUNCTION_TEMPLATE)
