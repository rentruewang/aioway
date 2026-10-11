# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect

import pytest
from torch import nn

from aioway._utils import Param, ParamKind, Sign


def int_add(a: int, b: int):
    return a + b


def float_add(a: float, b: float):
    return a + b


def add(a, b):
    return a + b


def greet(
    name: str, /, greeting: str = "hi", *args, loud: bool = False, **kwargs
) -> str:
    return f"{greeting} {name}"


class AddModule(nn.Module):
    def forward(self, x: int, y: int = 0) -> int:
        return x + y


@pytest.fixture
def add_sign_i():
    return Sign.from_callable(int_add)


@pytest.fixture
def add_sign_f():
    return Sign.from_callable(float_add)


@pytest.fixture
def add_sign():
    return Sign.from_callable(add)


@pytest.fixture
def greet_sign():
    return Sign.from_callable(greet)


# ParamKind


@pytest.mark.parametrize("kind", list(ParamKind))
def test_param_kind_round_trip(kind: ParamKind):
    assert ParamKind.from_inspect(kind.to_inspect()) is kind


@pytest.mark.parametrize("kind", list(ParamKind))
def test_param_kind_eq_inspect(kind: ParamKind):
    assert kind == kind.to_inspect()
    assert kind.to_inspect() == kind
    assert not kind != kind.to_inspect()


def test_param_kind_ne_other_kind():
    assert ParamKind.KEYWORD_ONLY != ParamKind.VAR_KEYWORD
    assert ParamKind.KEYWORD_ONLY != inspect.Parameter.VAR_KEYWORD


def test_param_kind_eq_str():
    assert ParamKind.KEYWORD_ONLY == "keyword_only"
    assert ParamKind.KEYWORD_ONLY == "KEYWORD_ONLY"
    assert ParamKind.KEYWORD_ONLY != "var_keyword"


def test_param_kind_hashable():
    assert len(set(ParamKind)) == 5


# Param


def test_param_build_defaults():
    param = Param.build("x")

    assert param == inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    assert param.kind is ParamKind.POSITIONAL_OR_KEYWORD
    assert param.default is inspect.Parameter.empty
    assert param.is_any_type


def test_param_build_all_fields():
    param = Param.build("flag", ParamKind.KEYWORD_ONLY, default=False, annotation=bool)

    assert param.name == "flag"
    assert param.is_keyword_only
    assert param.default is False
    assert param.annotation is bool
    assert not param.is_any_type


def test_param_build_invalid_name():
    with pytest.raises(ValueError):
        Param.build("not valid")


def test_param_kind_predicates(greet_sign: Sign):
    name, greeting, args, loud, kwargs = greet_sign.param_list

    assert name.is_positional_only
    assert greeting.is_positional_or_keyword
    assert args.is_var_positional
    assert loud.is_keyword_only
    assert kwargs.is_var_keyword


def test_param_eq():
    parameter = inspect.Parameter("x", inspect.Parameter.KEYWORD_ONLY)

    assert Param(parameter) == Param(parameter)
    assert Param(parameter) == parameter
    assert parameter == Param(parameter)
    assert Param(parameter) != "x"


def test_param_hashable():
    assert len({Param.build("x"), Param.build("x"), Param.build("y")}) == 2


def test_param_strip_type():
    param = Param.build("x", default=1, annotation=int)

    assert param.strip_type() == Param.build("x", default=1)


# Sign


def test_sign_hashable(add_sign_i, add_sign_f, add_sign):
    # If it can be put into `set`, it is hashable.
    signs = {add_sign_i, add_sign_f, add_sign}

    assert len(signs) == 3


def test_sign_not_equal(add_sign_i: Sign, add_sign_f: Sign, add_sign: Sign):
    assert add_sign_i != add_sign_f
    assert add_sign_i != add_sign
    assert add_sign_f != add_sign


def test_sign_outline_equal(add_sign_i: Sign, add_sign_f: Sign, add_sign: Sign):
    assert add_sign_i.strip_type() == add_sign_f.strip_type() == add_sign.strip_type()


def test_sign_eq_inspect_signature(greet_sign: Sign):
    assert greet_sign == inspect.signature(greet)


def test_sign_params(greet_sign: Sign):
    assert list(greet_sign.params) == ["name", "greeting", "args", "loud", "kwargs"]
    assert greet_sign.param_list == list(greet_sign.params.values())
    assert greet_sign.argc == 5


def test_sign_return_annotation(greet_sign: Sign, add_sign: Sign):
    assert greet_sign.return_annotation is str
    assert not greet_sign.returns_any_type
    assert add_sign.returns_any_type


def test_sign_strip_type_keeps_defaults(greet_sign: Sign):
    stripped = greet_sign.strip_type()

    assert all(param.is_any_type for param in stripped.param_list)
    assert stripped.returns_any_type
    assert stripped.params["greeting"].default == "hi"


def test_sign_from_inputs(add_sign: Sign):
    assert Sign.from_inputs(["a", "b"]) == add_sign


def test_sign_apply_fills_defaults(greet_sign: Sign):
    assert greet_sign.apply("bob") == {
        "name": "bob",
        "greeting": "hi",
        "args": (),
        "loud": False,
        "kwargs": {},
    }


def test_sign_bind_rejects_bad_args(add_sign: Sign):
    with pytest.raises(TypeError):
        add_sign.bind(1, 2, 3)


def test_sign_drop_first(add_sign: Sign):
    assert list(add_sign.drop_first().params) == ["b"]


def test_sign_args_annot(add_sign_i: Sign, add_sign_f: Sign, add_sign: Sign):
    # Missing annotations match anything.
    assert add_sign_i.args_annot() == add_sign.args_annot()
    assert add_sign_i.args_annot() != add_sign_f.args_annot()


def test_sign_args_annot_names_must_match(add_sign: Sign):
    assert add_sign.args_annot() != Sign.from_inputs(["a", "c"]).args_annot()


@pytest.mark.parametrize("module", [AddModule, AddModule()], ids=["type", "instance"])
def test_sign_from_nn_forward(module):
    sign = Sign.from_nn_forward(module)

    assert list(sign.params) == ["x", "y"]
    assert sign.return_annotation is int


def test_sign_from_nn_forward_rejects_non_module():
    with pytest.raises(TypeError):
        Sign.from_nn_forward(int)  # type: ignore[arg-type]
