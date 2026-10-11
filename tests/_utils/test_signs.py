# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect

import pytest
from torch import nn

from aioway._utils import Param, ParamKind, Sign


def _add_i(a: int, b: int):
    return a + b


def _add_f(a: float, b: float):
    return a + b


def _add(a, b):
    return a + b


def _greet(
    name: str, /, greeting: str = "hi", *args, loud: bool = False, **kwargs
) -> str:
    return f"{greeting} {name}"


class AddModule(nn.Module):
    def forward(self, x: int, y: int = 0) -> int:
        return x + y


class TestSign:
    @pytest.fixture(scope="class")
    @classmethod
    def add_i(cls):
        return Sign.from_callable(_add_i)

    @pytest.fixture(scope="class")
    @classmethod
    def add_f(cls):
        return Sign.from_callable(_add_f)

    @pytest.fixture(scope="class")
    @classmethod
    def add(cls):
        return Sign.from_callable(_add)

    @pytest.fixture(scope="class")
    @classmethod
    def greet(cls):
        return Sign.from_callable(_greet)

    def test_hashable(self, add_i, add_f, add):
        # If it can be put into `set`, it is hashable.
        signs = {add_i, add_f, add}

        assert len(signs) == 3

    def test_not_equal(self, add_i: Sign, add_f: Sign, add: Sign):
        assert add_i != add_f
        assert add_i != add
        assert add_f != add

    def test_outline_equal(self, add_i: Sign, add_f: Sign, add: Sign):
        assert add_i.strip_type() == add_f.strip_type() == add.strip_type()

    def test_eq_inspect_signature(self, greet: Sign):
        assert greet == inspect.signature(_greet)

    def test_params(self, greet: Sign):
        assert list(greet.params) == ["name", "greeting", "args", "loud", "kwargs"]
        assert greet.param_list == list(greet.params.values())
        assert greet.argc == 5

    def test_return_annotation(self, greet: Sign, add: Sign):
        assert greet.return_annotation is str
        assert not greet.returns_any_type
        assert add.returns_any_type

    def test_strip_type_keeps_defaults(self, greet: Sign):
        stripped = greet.strip_type()

        assert all(param.is_any_type for param in stripped.param_list)
        assert stripped.returns_any_type
        assert stripped.params["greeting"].default == "hi"

    def test_from_inputs(self, add: Sign):
        assert Sign.from_inputs(["a", "b"]) == add

    def test_apply_fills_defaults(self, greet: Sign):
        assert greet.apply("bob") == {
            "name": "bob",
            "greeting": "hi",
            "args": (),
            "loud": False,
            "kwargs": {},
        }

    def test_bind_rejects_bad_args(self, add: Sign):
        with pytest.raises(TypeError):
            add.bind(1, 2, 3)

    def test_drop_first(self, add: Sign):
        assert list(add.drop_first().params) == ["b"]

    def test_args_annot(self, add_i: Sign, add_f: Sign, add: Sign):
        # Missing annotations match anything.
        assert add_i.args_annot() == add.args_annot()
        assert add_i.args_annot() != add_f.args_annot()

    def test_args_annot_names_must_match(self, add: Sign):
        assert add.args_annot() != Sign.from_inputs(["a", "c"]).args_annot()

    @pytest.mark.parametrize(
        "module", [AddModule, AddModule()], ids=["type", "instance"]
    )
    def test_from_nn_forward(self, module):
        sign = Sign.from_nn_forward(module)

        assert list(sign.params) == ["x", "y"]
        assert sign.return_annotation is int

    def test_nn_forward_only_nn(self):
        with pytest.raises(TypeError):
            Sign.from_nn_forward(int)  # type: ignore[arg-type]

    def test_predicates(self, greet: Sign):
        name, greeting, args, loud, kwargs = greet.param_list

        assert name.is_positional_only
        assert greeting.is_positional_or_keyword
        assert args.is_var_positional
        assert loud.is_keyword_only
        assert kwargs.is_var_keyword


class TestParamKind:

    @pytest.mark.parametrize("kind", list(ParamKind))
    def test_round_trip(self, kind: ParamKind):
        assert ParamKind.from_inspect(kind.to_inspect()) is kind

    @pytest.mark.parametrize("kind", list(ParamKind))
    def test_eq_inspect(self, kind: ParamKind):
        assert kind == kind.to_inspect()
        assert kind.to_inspect() == kind
        assert not kind != kind.to_inspect()

    def test_ne_other_kind(self):
        assert ParamKind.KEYWORD_ONLY != ParamKind.VAR_KEYWORD
        assert ParamKind.KEYWORD_ONLY != inspect.Parameter.VAR_KEYWORD

    def test_eq_str(self):
        assert ParamKind.KEYWORD_ONLY == "keyword_only"
        assert ParamKind.KEYWORD_ONLY == "KEYWORD_ONLY"
        assert ParamKind.KEYWORD_ONLY != "var_keyword"

    def test_hashable(self):
        assert len(set(ParamKind)) == 5


class TestParam:

    def test_build_defaults(self):
        param = Param.build("x")

        assert param == inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        assert param.kind is ParamKind.POSITIONAL_OR_KEYWORD
        assert param.default is inspect.Parameter.empty
        assert param.is_any_type

    def test_build_all_fields(self):
        param = Param.build(
            "flag", ParamKind.KEYWORD_ONLY, default=False, annotation=bool
        )

        assert param.name == "flag"
        assert param.is_keyword_only
        assert param.default is False
        assert param.annotation is bool
        assert not param.is_any_type

    def test_build_invalid_name(self):
        with pytest.raises(ValueError):
            Param.build("not valid")

    def test_eq(self):
        parameter = inspect.Parameter("x", inspect.Parameter.KEYWORD_ONLY)

        assert Param(parameter) == Param(parameter)
        assert Param(parameter) == parameter
        assert parameter == Param(parameter)
        assert Param(parameter) != "x"

    def test_hashable(self):
        assert len({Param.build("x"), Param.build("x"), Param.build("y")}) == 2

    def test_strip_type(self):
        param = Param.build("x", default=1, annotation=int)

        assert param.strip_type() == Param.build("x", default=1)
