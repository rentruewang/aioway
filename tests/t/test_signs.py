# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
from collections import abc as cabc

import pytest
import torch
from torch import nn
from torch.utils import _pytree as pyt

from aioway.t import BoundArgs, Param, ParamKind, Sign


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


@pytest.fixture(scope="class")
def add():
    return Sign.from_callable(_add)


@pytest.fixture(scope="class")
def greet():
    return Sign.from_callable(_greet)


@pytest.fixture(scope="class")
def add_i():
    return Sign.from_callable(_add_i)


@pytest.fixture(scope="class")
def add_f():
    return Sign.from_callable(_add_f)


class TestSign:

    @pytest.fixture(params=[torch.sum, torch.add, torch.max])
    def torch_func(self, request: pytest.FixtureRequest):
        return request.param

    def test_torch_functions(self, torch_func: cabc.Callable):
        sign = Sign.from_callable(torch_func)
        assert isinstance(sign, Sign)
        assert isinstance(sign.to_inspect(), inspect.Signature)

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

    def test_apply_defaults(self, greet: Sign):
        assert greet.apply("bob") == {
            "name": "bob",
            "greeting": "hi",
            "args": (),
            "loud": False,
            "kwargs": {},
        }

    def test_bad_args(self, add: Sign):
        with pytest.raises(TypeError):
            add.bind(1, 2, 3)

    def test_drop_first(self, add: Sign):
        assert list(add.drop_first().params) == ["b"]

    def test_args_annot(self, add_i: Sign, add_f: Sign, add: Sign):
        # Missing annotations match anything.
        assert add_i.args_annot() == add.args_annot()
        assert add_i.args_annot() != add_f.args_annot()

    def test_name_match(self, add: Sign):
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


class TestBoundArgs:

    @pytest.fixture(scope="class")
    @classmethod
    def bound(cls, greet: Sign):
        return greet.bind("bob", "hey", 1, 2, loud=True, extra=3)

    def test_bind(self, bound: BoundArgs):
        assert isinstance(bound, BoundArgs)
        assert isinstance(bound.to_inspect(), inspect.BoundArguments)
        assert bound.arguments == {
            "name": "bob",
            "greeting": "hey",
            "args": (1, 2),
            "loud": True,
            "kwargs": {"extra": 3},
        }

    def test_apply_defaults(self, greet: Sign):
        bound = greet.bind("bob")
        bound = bound.apply_defaults()
        assert bound.arguments == {
            "name": "bob",
            "greeting": "hi",
            "args": (),
            "loud": False,
            "kwargs": {},
        }
        assert greet.apply("bob") == bound.arguments

    def test_args_kwargs(self, bound: BoundArgs):
        assert bound.args == ("bob", "hey", 1, 2)
        assert bound.kwargs == {"loud": True, "extra": 3}
        assert _greet(*bound.args, **bound.kwargs) == "hey bob"

    def test_sign(self, greet: Sign, bound: BoundArgs):
        assert isinstance(bound.sign, Sign)
        assert bound.sign == greet
        assert bound.sign.to_inspect() == greet.to_inspect()

        # Binding the same call through `.sign` gives back the same `BoundArgs`.
        assert bound.sign.bind(*bound.args, **bound.kwargs) == bound

    def test_mapping_like(self, bound: BoundArgs):
        assert len(bound) == 5
        assert list(bound.keys()) == ["name", "greeting", "args", "loud", "kwargs"]
        assert bound["name"] == "bob"
        assert {name: bound[name] for name in bound.keys()} == bound.arguments

    def test_eq(self, greet: Sign, bound: BoundArgs):
        assert bound == greet.bind("bob", "hey", 1, 2, loud=True, extra=3)
        assert bound == bound.to_inspect()
        assert bound.to_inspect() == bound
        assert bound == bound.arguments
        assert bound != greet.bind("bob")
        assert bound != "whatev"

    def test_repr_str(self, bound: BoundArgs):
        assert repr(bound) == repr(bound.to_inspect())
        assert str(bound) == str(bound.to_inspect())

    def test_tensors(self, add: Sign):
        a, b = torch.ones(2), torch.zeros(2)
        bound = add.bind(a, b)

        assert [id(t) for t in bound.tensors] == [id(a), id(b)]
        assert bound.tensors == bound.tensors

    def test_tree_flatten(self, greet: Sign, bound: BoundArgs):
        leaves, spec = pyt.tree_flatten(bound)

        # The `*args` tuple and the `**kwargs` dict are flattened as well.
        assert leaves == ["bob", "hey", 1, 2, True, 3]

        rebuilt = pyt.tree_unflatten(leaves, spec)
        assert isinstance(rebuilt, BoundArgs)
        assert rebuilt == bound
        assert rebuilt.sign == greet

    def test_tree_flatten_with_path(self, bound: BoundArgs):
        flat, _ = pyt.tree_flatten_with_path(bound)

        assert [pyt.keystr(path) for path, _ in flat] == [
            "['name']",
            "['greeting']",
            "['args'][0]",
            "['args'][1]",
            "['loud']",
            "['kwargs']['extra']",
        ]

    def test_tree_map(self, add: Sign):
        bound = add.bind(torch.ones(2), b=torch.ones(2))
        doubled = pyt.tree_map(lambda t: t * 2, bound)

        assert isinstance(doubled, BoundArgs)
        assert doubled.sign == add
        assert torch.equal(_add(*doubled.args, **doubled.kwargs), torch.full([2], 4.0))

    def test_tree_map_many(self, greet: Sign):
        # Once defaults are applied, the same parameters are always bound,
        # and calls that differ only in values share one tree structure.
        first = greet.bind("bob", loud=True).apply_defaults()
        second = greet.bind("eve", "yo").apply_defaults()

        assert pyt.tree_structure(first) == pyt.tree_structure(second)

        picked = pyt.tree_map(lambda x, y: (x, y), first, second)
        assert picked.arguments["name"] == ("bob", "eve")
        assert picked.arguments["loud"] == (True, False)

    def test_bind_no_default(self, greet: Sign):
        bound = greet.bind("bob")
        assert bound.arguments == {"name": "bob"}

        assert bound.apply_defaults() is not bound
        assert bound.arguments == greet.apply("bob")

    def test_ne_dict(self, bound: BoundArgs):
        assert bound != {"name": "bob", "greeting": "bye"}

    def test_tensors_nested(self, greet: Sign):
        a, b = torch.ones(1), torch.ones(2)
        bound = greet.bind("bob", "hey", a, extra=b)

        assert [id(t) for t in bound.tensors] == [id(a), id(b)]

    def test_tensors_apply_defaults(self):
        x, w = torch.ones(2), torch.ones(2)
        bound = Sign.from_callable(lambda x, w=w: x * w).bind(x)
        assert len(bound.tensors) == 1

        bound = bound.apply_defaults()
        assert len(bound.tensors) == 2


class TestParamKind:

    @pytest.mark.parametrize("kind", list(ParamKind))
    def test_round_trip(self, kind: ParamKind):
        assert ParamKind.from_inspect(kind.to_inspect()) == kind

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
        assert param.default_is_empty
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
