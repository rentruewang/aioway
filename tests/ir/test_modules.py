# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway._utils import Stack
from aioway.ir import ModuleHist, ModuleThunk, ModuleTracker


class Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class Wrapped(nn.Module):
    """A parent whose output is a fresh tensor, not its child's."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = Double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1


def _make_thunk(
    module: nn.Module,
    *args: torch.Tensor,
    result: object = None,
    parents: tuple[nn.Module, ...] = (),
) -> ModuleThunk:
    return ModuleThunk(
        func=module,
        args=args,
        kwargs={},
        result=module(*args) if result is None else result,
        parents=parents,
    )


@pytest.fixture
def module() -> nn.Module:
    return Double()


@pytest.fixture
def thunk(module: nn.Module) -> ModuleThunk:
    return _make_thunk(module, torch.ones(3))


@pytest.fixture
def hist() -> ModuleHist:
    return ModuleHist()


@pytest.fixture
def tracker(hist: ModuleHist) -> ModuleTracker:
    return ModuleTracker(stack=Stack(), hist=hist)


def test_thunk_keeps_parents(module: nn.Module) -> None:
    parent = Wrapped()
    thunk = _make_thunk(module, torch.ones(3), parents=(parent,))

    assert thunk.parents == (parent,)


def test_thunk_upstream_yields(module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = _make_thunk(module, x, y, result=x)

    assert thunk.inputs == {x, y}


def test_thunk_downstream_yields(thunk: ModuleThunk) -> None:
    assert thunk.outputs == [thunk.result]


def test_hist_starts_empty(hist: ModuleHist) -> None:
    assert len(hist) == 0


def test_append_stores_thunk(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    assert len(hist) == 1
    assert hist[0] is thunk


def test_thunk_looks_up(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    assert hist.thunk_of(thunk.outputs[0]) is thunk


def test_append_index_all_outputs(hist: ModuleHist, module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = _make_thunk(module, result=(x, y))

    hist.append(thunk)

    assert hist.thunk_of(x) is thunk
    assert hist.thunk_of(y) is thunk


def test_append_fail_dups(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    with pytest.raises(KeyError):
        hist.append(thunk)


def test_thunk_of_unknown_fail(hist: ModuleHist, thunk: ModuleThunk) -> None:
    with pytest.raises(KeyError):
        hist.thunk_of(thunk.outputs[0])


def test_hook_appends_thunk(hist: ModuleHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)

    hist.module_forward_hook(module, (x,), y)

    (recorded,) = hist.history
    assert recorded.func is module
    assert list(recorded.inputs) == [x]
    assert list(recorded.outputs) == [y]
    assert recorded.parents == ()


def test_tracker_yield(tracker: ModuleTracker) -> None:
    with tracker() as yielded:
        assert yielded is tracker


def test_tracker_records_leaves(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = Wrapped()

    with tracker():
        model(torch.ones(3))

    assert [t.func for t in hist.history] == [model.inner]


def test_tracker_records_parents(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = Wrapped()

    with tracker():
        model(torch.ones(3))

    assert hist[0].parents[0] == model


def test_tracker_records_shapes(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = nn.Sequential(nn.Linear(4, 2))

    with tracker():
        model(torch.ones(3, 4))

    (thunk,) = hist.history
    assert thunk.inputs[0].shape == (3, 4)
    assert thunk.outputs[0].shape == (3, 2)


def test_tracker_empty_stack(tracker: ModuleTracker) -> None:
    with tracker():
        Wrapped()(torch.ones(3))

    assert not tracker.stack


def test_tracker_active_only_context(tracker: ModuleTracker, hist: ModuleHist) -> None:
    with tracker():
        pass

    Wrapped()(torch.ones(3))

    assert len(hist) == 0
