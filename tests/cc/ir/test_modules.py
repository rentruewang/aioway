# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from torch import nn

from aioway._utils import Stack
from aioway.cc import ModuleHist, ModuleThunk, ModuleTracker


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


def make_thunk(
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
    return make_thunk(module, torch.ones(3))


@pytest.fixture
def hist() -> ModuleHist:
    return ModuleHist()


@pytest.fixture
def tracker(hist: ModuleHist) -> ModuleTracker:
    return ModuleTracker(stack=Stack(), hist=hist)


# --- ModuleThunk --------------------------------------------------------------


def test_thunk_keeps_parents(module: nn.Module) -> None:
    parent = Wrapped()
    thunk = make_thunk(module, torch.ones(3), parents=(parent,))

    assert thunk.parents == (parent,)


def test_thunk_upstream_yields_args(module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = make_thunk(module, x, y, result=x)

    assert list(thunk.upstream()) == [x, y]


def test_thunk_downstream_yields_result(thunk: ModuleThunk) -> None:
    assert list(thunk.downstream()) == [thunk.result]


# --- ModuleHist ---------------------------------------------------------------


def test_hist_starts_empty(hist: ModuleHist) -> None:
    assert len(hist) == 0


def test_append_stores_thunk(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    assert len(hist) == 1
    assert hist[0] is thunk


def test_thunk_of_looks_up_by_output(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    assert hist.thunk_of(next(thunk.downstream())) is thunk


def test_append_indexes_every_output(hist: ModuleHist, module: nn.Module) -> None:
    x, y = torch.ones(3), torch.zeros(3)
    thunk = make_thunk(module, result=(x, y))

    hist.append(thunk)

    assert hist.thunk_of(x) is thunk
    assert hist.thunk_of(y) is thunk


def test_append_conflicting_output_raises(hist: ModuleHist, thunk: ModuleThunk) -> None:
    hist.append(thunk)

    with pytest.raises(KeyError):
        hist.append(thunk)


def test_thunk_of_unknown_output_raises(hist: ModuleHist, thunk: ModuleThunk) -> None:
    with pytest.raises(KeyError):
        hist.thunk_of(next(thunk.downstream()))


def test_forward_hook_appends_thunk(hist: ModuleHist, module: nn.Module) -> None:
    x = torch.ones(3)
    y = module(x)

    hist.module_forward_hook(module, (x,), y)

    (recorded,) = hist.history
    assert recorded.func is module
    assert list(recorded.upstream()) == [x]
    assert list(recorded.downstream()) == [y]
    assert recorded.parents == ()


# --- ModuleTracker ------------------------------------------------------------


def test_tracker_yields_self(tracker: ModuleTracker) -> None:
    with tracker() as yielded:
        assert yielded is tracker


def test_tracker_records_only_leaves(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = Wrapped()

    with tracker():
        model(torch.ones(3))

    assert [t.func for t in hist.history] == [model.inner]


def test_tracker_records_parent_stack(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = Wrapped()

    with tracker():
        model(torch.ones(3))

    assert hist[0].parents[0] == model


def test_tracker_records_shapes(tracker: ModuleTracker, hist: ModuleHist) -> None:
    model = nn.Sequential(nn.Linear(4, 2))

    with tracker():
        model(torch.ones(3, 4))

    (thunk,) = hist.history
    assert next(thunk.upstream()).shape == (3, 4)
    assert next(thunk.downstream()).shape == (3, 2)


def test_tracker_unwinds_its_stack(tracker: ModuleTracker) -> None:
    with tracker():
        Wrapped()(torch.ones(3))

    assert list(tracker.stack) == []


def test_tracker_stops_recording_after_exit(
    tracker: ModuleTracker, hist: ModuleHist
) -> None:
    with tracker():
        pass

    Wrapped()(torch.ones(3))

    assert len(hist) == 0
