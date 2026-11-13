# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import inspect
import typing

import pytest
import torch
from torch import nn
from torch import testing as tt

from aioway._signs import Param, Sign
from aioway.ops import (
    DagOpExec,
    FuncInputOp,
    LazyCacheOpExec,
    ModuleThunkOp,
    Op,
    OpFunc,
    ValueOp,
    default_op_exec,
    exec_op,
    executor,
    set_executor,
)


class SumCountOp(Op):
    "Sums its children (or returns `value` if none) and counts executions."

    value: int = 0
    children: list = dcls.field(default_factory=list)
    calls: int = 0

    @typing.override
    def execute(self):
        self.calls += 1
        return self.value + sum(exec_op(c) for c in self.children)

    @property
    @typing.override
    def __deps__(self):
        return self.children


type CachedExec = LazyCacheOpExec | DagOpExec


@pytest.fixture(params=[LazyCacheOpExec, DagOpExec], ids=lambda c: c.__name__)
def exec_cls(request: pytest.FixtureRequest) -> type[CachedExec]:
    "The executors that execute each `Op` at most once per evaluation."
    return request.param


def test_default_executor():
    assert executor() is default_op_exec


def test_default_no_cache():
    op = SumCountOp(1)
    op()
    op()
    assert op.calls == 2


def test_set_executor(exec_cls: type[CachedExec]):
    cache = exec_cls()
    with set_executor(cache) as got:
        assert got is cache
        assert executor() is cache
    assert executor() is default_op_exec


def test_set_executor_error(exec_cls: type[CachedExec]):
    cache = exec_cls()
    with pytest.raises(RuntimeError), set_executor(cache):
        raise RuntimeError
    assert executor() is default_op_exec


def test_cached_per_evaluation(exec_cls: type[CachedExec]):
    op = SumCountOp(1)
    with set_executor(exec_cls()):
        assert op() == 1
        assert op() == 1
    assert op.calls == 2


def test_lazy_cache_cleanup():
    cache = LazyCacheOpExec()
    with set_executor(cache):
        SumCountOp(1)()
    assert len(cache) == 0


def test_cached_shared_dep(exec_cls: type[CachedExec]):
    shared = SumCountOp(1)
    root = SumCountOp(0, [SumCountOp(2, [shared]), SumCountOp(3, [shared])])
    with set_executor(exec_cls()):
        assert root() == 7
    assert shared.calls == 1


def test_cache_op(exec_cls: type[CachedExec]):
    shared = SumCountOp(1)
    root = SumCountOp(0, [shared, shared])
    with exec_cls.enable() as cache:
        assert executor() is cache
        assert root() == 2
    assert executor() is default_op_exec
    assert shared.calls == 1


def test_dag_frees_results():
    dag = DagOpExec()
    alive: list[int] = []

    class SpyOp(SumCountOp):
        def execute(self):
            alive.append(len(dag._results))
            return super().execute()

    op = SpyOp(1)
    for _ in range(10):
        op = SpyOp(0, [op])

    with set_executor(dag):
        assert op() == 1

    # A chain only ever needs the previous result.
    assert max(alive) == 1
    assert len(dag._results) == 0


def test_dag_error_cleans_up():
    dag = DagOpExec()

    class FailOp(SumCountOp):
        def execute(self):
            raise RuntimeError

    op = SumCountOp(0, [SumCountOp(1), FailOp()])
    with set_executor(dag), pytest.raises(RuntimeError):
        op()

    assert len(dag._results) == 0


def inp(name: str) -> FuncInputOp:
    return FuncInputOp(param=Param.build(name), typ=int)


def test_func_signature():
    f = OpFunc(SumCountOp(0, [inp("a"), inp("b")]))
    assert list(inspect.signature(f).parameters) == ["a", "b"]


def test_func_call():
    f = OpFunc(SumCountOp(0, [inp("a"), SumCountOp(0, [inp("b"), inp("c")])]))
    assert f(1, 2, 3) == 6
    assert f(a=1, b=2, c=3) == 6


def test_func_dedup_input():
    x = inp("x")
    op = SumCountOp(0, [x, x])

    # (x, x) would have signature (x)
    assert OpFunc(op).signature == Sign.from_input_names(["x"])


def test_func_input_mismatch():
    x = inp("x")

    # Same name, different param.
    diff_param = SumCountOp(
        0, [x, FuncInputOp(param=Param.build("x", default=0), typ=int)]
    )
    with pytest.raises(TypeError, match="same name"):
        OpFunc(diff_param)

    # Same name, different typ.
    diff_typ = SumCountOp(0, [x, FuncInputOp(param=Param.build("x"), typ=float)])
    with pytest.raises(TypeError, match="same name"):
        OpFunc(diff_typ)


def test_func_bad_args():
    func = OpFunc(inp("a"))
    with pytest.raises(TypeError):
        func()


def test_func_no_stale_result(exec_cls: type[CachedExec]):
    f = OpFunc(SumCountOp(0, [inp("a"), inp("b")]))
    # This enable should have no effect.
    with exec_cls.enable():
        assert f(1, 2) == 3
        assert f(3, 4) == 7


def test_func_input_without_exec():
    x = inp("x")
    with pytest.raises(RuntimeError):
        x()


def test_func_module():
    layer = nn.Linear(2, 3)
    x = torch.randn(1, 2)
    f = OpFunc(ModuleThunkOp(func=layer, args=(inp("x"),), kwargs={}))

    f_params = set(f.parameters())
    layer_params = set(layer.parameters())
    assert f_params == layer_params
    tt.assert_close(f(x), layer(x))


def test_func_apply_equal_inputs():
    # Two equal, separate inputs.
    f = OpFunc(SumCountOp(0, [inp("x"), inp("x")]))
    g = f.apply(ValueOp(1))

    # Every `x` is bound, so nothing is left in the signature.
    assert g.signature == Sign.from_input_names([])
