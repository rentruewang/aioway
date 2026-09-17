# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway.nets import ModuleInOutHist, ModuleInOutThunk, capture_module_hist


class Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class AddOne(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = Double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1


@pytest.fixture
def module() -> nn.Module:
    return Double()


@pytest.fixture
def thunk(module: nn.Module) -> ModuleInOutThunk:
    x = torch.ones(3)
    return ModuleInOutThunk(module=module, input=(x,), output=module(x))


@pytest.fixture
def hist() -> ModuleInOutHist:
    return ModuleInOutHist()


def test_thunk_inputs(module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = ModuleInOutThunk(module=module, input=(x, y), output=x)
    assert _ids_set(*thunk.inputs()) == _ids_set(x, y)


def test_thunk_nested_inputs(module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = ModuleInOutThunk(module=module, input=(x, (y,)), output=x)
    assert _ids_set(*thunk.inputs()) == _ids_set(x, y)


def test_thunk_outputs(module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = ModuleInOutThunk(module=module, input=(), output=(x, y))

    assert _ids_set(*thunk.outputs()) == _ids_set(x, y)


def test_hist_starts_empty(hist: ModuleInOutHist) -> None:
    assert len(hist) == 0


def test_hist_append(hist: ModuleInOutHist, thunk: ModuleInOutThunk) -> None:
    hist.append(thunk)

    assert len(hist) == 1
    assert hist[0] is thunk


def test_thunk_by_output(hist: ModuleInOutHist, thunk: ModuleInOutThunk) -> None:
    hist.append(thunk)

    for out in thunk.outputs():
        assert hist.thunk_of(out) is thunk


def test_append_by_output_multi(hist: ModuleInOutHist, module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = ModuleInOutThunk(module=module, input=(), output=(x, y))

    hist.append(thunk)

    for out in thunk.outputs():
        assert hist.thunk_of(out) is thunk


def test_no_append_twice(hist: ModuleInOutHist, thunk: ModuleInOutThunk) -> None:
    hist.append(thunk)

    with pytest.raises(KeyError):
        hist.append(thunk)


def test_missing_thunk_lookup(hist: ModuleInOutHist, thunk: ModuleInOutThunk) -> None:
    with pytest.raises(KeyError):
        hist.thunk_of(next(thunk.outputs()))


def test_forward_hook_appends(hist: ModuleInOutHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)

    hist.module_forward_hook(module, (x,), y)

    [recorded] = hist.history
    assert recorded.module is module
    assert list(recorded.inputs()) == [x]
    assert list(recorded.outputs()) == [y]


def test_track_with_register(hist: ModuleInOutHist, module: nn.Module) -> None:
    with hist.register_module_forward_hook():
        module(ones := torch.ones(3))

    assert len(hist) == 1
    assert hist[0].module is module
    assert list(hist[0].inputs()) == [ones]


def test_no_track_outside_register(hist: ModuleInOutHist, module: nn.Module) -> None:
    with hist.register_module_forward_hook():
        pass

    module(torch.ones(3))
    assert len(hist) == 0


def test_capture_nested() -> None:
    model = AddOne()

    with capture_module_hist() as hist:
        model(torch.ones(3))

    assert [t.module for t in hist.history] == [model.inner, model]


def test_capture_keeps_shape() -> None:
    model = nn.Linear(4, 2)

    with capture_module_hist() as hist:
        model(torch.ones(3, 4))

    (thunk,) = hist.history
    assert {t.shape for t in thunk.inputs()} == {(3, 4)}
    assert {t.shape for t in thunk.outputs()} == {(3, 2)}


def _ids_set(*tensors: torch.Tensor) -> set[int]:
    return {id(t) for t in tensors}
