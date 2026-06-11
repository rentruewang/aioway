# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections import abc as cabc

import pytest
import tensordict as td

from aioway._torch import tdict_all_equal, tdict_rename
from aioway.hop import TdictHop, hop_dcls
from aioway.io import SourceListHop
from aioway.relalg import (
    ApplyHop,
    FuncFilterHop,
    MapHop,
    ProjectHop,
    RenameHop,
)


@dcls.dataclass
class SaveLastState:

    last: td.TensorDict = dcls.field(init=False, repr=False)
    "The last batch."


@hop_dcls
class SaveLastMapStream(MapHop):
    "`Stream` that saves the last `__next__` call."

    @typing.override
    def _apply(self, batch: td.TensorDict) -> td.TensorDict:
        self.state.last = batch
        return batch

    @property
    def last(self) -> td.TensorDict:
        return self.state.last

    @functools.cached_property
    def state(self):
        return SaveLastState()


@pytest.fixture
def save_last(table_stream: TdictHop):
    "The stream that is wrapped, preserving the last item."

    return SaveLastMapStream(table_stream)


@pytest.fixture
def map_stream(request: pytest.FixtureRequest, save_last: SaveLastMapStream):
    "Indirect fixture to create `MapStream`s based on a builder function."

    builder: cabc.Callable[[TdictHop], MapHop] = request.param

    if not callable(builder):
        raise TypeError("The `map_stream` fixture only accepts function parameters.")

    return typing.cast(MapHop, builder(save_last))


def _pred_filter_builder(source):
    return FuncFilterHop(
        source=source,
        predicate=lambda t: (t["f1d"] > 0),
    )


@pytest.mark.parametrize("map_stream", [_pred_filter_builder], indirect=True)
def test_filter(map_stream: TdictHop, save_last: SaveLastMapStream):
    "Testing the 2 filter streams and whether they are doing their jobs."

    for filtered in map_stream:
        f1d = save_last.last["f1d"]
        manual_filtered: td.TensorDict = save_last.last[f1d > 0]
        assert filtered.shape == manual_filtered.shape, {
            "lhs.shape": filtered.shape,
            "rhs.shape": manual_filtered.shape,
        }
        assert tdict_all_equal(filtered, manual_filtered)


def _rename_builder(save_last: SaveLastMapStream):
    renames = {"f1d": "f1", "f2d": "f2", "i1d": "i1", "i2d": "i2"}
    return RenameHop(source=save_last, renames=renames)


@pytest.mark.parametrize("map_stream", [_rename_builder], indirect=True)
def test_rename(map_stream: TdictHop, save_last: SaveLastMapStream):
    "Testing the renaming functionality."

    for renamed in map_stream:
        manual_renamed = tdict_rename(
            save_last.last, f1d="f1", f2d="f2", i1d="i1", i2d="i2"
        )
        assert tdict_all_equal(renamed, manual_renamed)


def _apply_builder(save_last: SaveLastMapStream):

    func = lambda td: tdict_rename(td, f1d="f", i1d="i")
    schema = lambda attrs: attrs.rename(f1d="f", i1d="i")
    return ApplyHop(source=save_last, apply=func, schema=schema)


@pytest.mark.parametrize("map_stream", [_apply_builder], indirect=True)
def test_apply(map_stream: ApplyHop, save_last: SaveLastMapStream):
    for mapped in map_stream:
        assert tdict_all_equal(mapped, map_stream.apply(save_last.last))


def _project_builder(save_last: SaveLastMapStream):
    return ProjectHop(source=save_last, subset=["f1d", "i2d"])


@pytest.mark.parametrize("map_stream", [_project_builder], indirect=True)
def test_project(map_stream: TdictHop, save_last: SaveLastMapStream):
    for projected in map_stream:
        assert tdict_all_equal(projected, save_last.last.select("f1d", "i2d"))


@pytest.mark.parametrize(
    "map_stream",
    [
        _pred_filter_builder,
        _rename_builder,
        _apply_builder,
        _project_builder,
    ],
    indirect=True,
)
def test_map_stream_one_to_one(map_stream: MapHop, save_last: SaveLastMapStream):
    map_stream_iter = iter(map_stream)

    assert (
        map_stream.source is save_last
    ), f"Malformed input {map_stream}, should have source={save_last}"

    assert map_stream_iter.idx == 0, "Pre iteration stream's index starts with 0."

    for idx, _ in enumerate(map_stream_iter, start=1):
        # Ensure that the input is also exhausted.
        assert idx == map_stream_iter.idx

    assert map_stream_iter.idx == save_last.size


@pytest.mark.parametrize(
    "map_stream", [_project_builder, _apply_builder], indirect=True
)
def test_caching(map_stream: TdictHop):
    cached = SourceListHop.exhaust(map_stream)
    assert cached.size == map_stream.size
