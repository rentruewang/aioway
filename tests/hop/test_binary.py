# Copyright (c) AIoWay Authors - All Rights Reserved

import collections
from collections import abc as cabc

import pytest
import tensordict as td
import torch

from aioway._utils import tdict_all_equal
from aioway.dsets import TdictListHop
from aioway.hop import NestedLoopJoinIter, TdictIter, ZipIter


@pytest.fixture
def lhs_stream(concat_stream: TdictIter) -> TdictListHop:
    return TdictListHop(list(concat_stream))


@pytest.fixture
def rhs_stream(joinable_stream: TdictIter) -> TdictListHop:
    return TdictListHop(list(joinable_stream))


def test_lhs_stream_length(concat_stream: TdictIter, lhs_stream: TdictIter):
    assert concat_stream.size == lhs_stream.size


def test_rhs_stream_length(joinable_stream: TdictIter, rhs_stream: TdictListHop):
    assert joinable_stream.size == rhs_stream.size


@pytest.fixture
def binary_stream(
    request: pytest.FixtureRequest,
    lhs_stream: TdictIter,
    rhs_stream: TdictListHop,
):
    "An indirect fixture that takes in a builder function and outputs a stream."

    builder: cabc.Callable[[TdictIter, TdictIter], TdictIter] = request.param

    if not callable(builder):
        raise TypeError("Indirect fixture `binary_stream` only accepts functions.")

    result = builder(lhs_stream, rhs_stream)
    assert isinstance(result, TdictIter)
    return result


def _zip_builder(lhs_stream: TdictIter, rhs_stream: TdictListHop):
    return ZipIter(left=lhs_stream, right=rhs_stream)


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip_input_len(
    binary_stream: TdictIter,
    concat_stream: TdictIter,
    rhs_stream: TdictListHop,
):
    assert min(concat_stream.size, rhs_stream.size) == binary_stream.size


@pytest.mark.parametrize("binary_stream", [_zip_builder], indirect=True)
def test_zip(
    binary_stream: ZipIter,
    lhs_stream: TdictListHop,
    rhs_stream: TdictListHop,
):
    assert isinstance(binary_stream, ZipIter)
    assert isinstance(lhs_stream, TdictListHop)
    assert isinstance(rhs_stream, TdictListHop)

    binary_stream_iter = iter(binary_stream)

    assert not binary_stream_iter.started

    assert binary_stream.left is lhs_stream
    assert binary_stream.right is rhs_stream
    for result in binary_stream_iter:
        concat = td.merge_tensordicts(
            lhs_stream.sequence[binary_stream_iter.idx - 1],
            rhs_stream.sequence[binary_stream_iter.idx - 1],
        )
        assert tdict_all_equal(result, concat)


def _join_builder(lhs_stream: TdictIter, rhs_stream: TdictListHop):
    return NestedLoopJoinIter(left=lhs_stream, right=rhs_stream, key="i1d")


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_input_len(
    binary_stream: TdictIter,
    lhs_stream: TdictIter,
    rhs_stream: TdictListHop,
):
    assert binary_stream.size == lhs_stream.size * rhs_stream.size


@pytest.mark.parametrize(
    "to_slice",
    [
        lambda x: [x],
        lambda t: [t[0:2], t[2:4]],
        lambda t: [t[[1, 3]], t[[0, 2]]],
    ],
)
def test_simple_nested_loop_join(
    to_slice: cabc.Callable[[td.TensorDict], list[td.TensorDict]],
):
    left = td.TensorDict(
        {
            "a": torch.tensor([1, 3, 2, 2]),
            "b": torch.tensor([4, 10, 5, 6]),
        }
    ).auto_batch_size_()
    right = td.TensorDict(
        {
            "a": torch.tensor([1, 3, 2, 2]),
            "c": torch.tensor([7, 11, 8, 9]),
        }
    ).auto_batch_size_()

    left_stream = TdictListHop(to_slice(left))
    right_stream = TdictListHop(to_slice(right))

    out = td.cat(list(NestedLoopJoinIter(left_stream, right_stream, key="a")))

    def sort_by_abc(td: td.TensorDict):
        for key in "cba":
            indices = torch.argsort(td[key], stable=True)
            td = td[indices]
        return td

    assert tdict_all_equal(
        sort_by_abc(out),
        td.TensorDict(
            {
                "a": [1, 2, 2, 2, 2, 3],
                "b": [4, 5, 5, 6, 6, 10],
                "c": [7, 8, 9, 8, 9, 11],
            }
        ),
    )


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_join_equal_as_original(
    binary_stream: TdictIter,
    lhs_stream: TdictIter,
    rhs_stream: TdictListHop,
):
    block_frame_block = td.cat(list(lhs_stream))
    joinable_frame_block = td.cat(list(rhs_stream))

    # Performing the join here.
    results: list[td.TensorDict] = list(binary_stream)
    assert len(results), "The binary stream is empty."
    answer_items = td.cat(results)["i1d"]

    # Do it at once, using `ListStream` as it yields everything in 1 batch.
    ground_truth = td.cat(
        list(
            NestedLoopJoinIter(
                left=TdictListHop([block_frame_block]),
                right=TdictListHop([joinable_frame_block]),
                key="i1d",
            )
        )
    )

    answer_count = collections.Counter(answer_items.tolist())
    truth_count = collections.Counter(ground_truth["i1d"].tolist())

    assert answer_count == truth_count


@pytest.mark.parametrize("binary_stream", [_join_builder], indirect=True)
def test_match_functionally(
    binary_stream: TdictIter,
    lhs_stream: TdictIter,
    rhs_stream: TdictListHop,
):
    block_frame_block = td.cat(list(lhs_stream))
    joinable_frame_block = td.cat(list(rhs_stream))

    # Performing the join here.
    results = list(binary_stream)
    answer_items = td.cat(results)["i1d"]

    answer_count = collections.Counter(answer_items.tolist())

    left_count = collections.Counter(block_frame_block["i1d"].tolist())
    right_count = collections.Counter(joinable_frame_block["i1d"].tolist())

    # Functionally correct join.
    assert left_count.keys() == {*block_frame_block["i1d"].tolist()}
    assert right_count.keys() == {*joinable_frame_block["i1d"].tolist()}
    assert answer_count.keys() == left_count.keys() & right_count.keys()

    for key in answer_count.keys():
        assert answer_count[key] == left_count[key] * right_count[key]


@pytest.mark.parametrize(
    "binary_stream",
    [_zip_builder, _join_builder],
    indirect=True,
)
def test_binary_stream_in_list(
    binary_stream: NestedLoopJoinIter | ZipIter,
):
    binary_stream_iter = iter(binary_stream)
    assert binary_stream.size, "The binary stream is empty."
    assert binary_stream_iter.idx == 0, "Pre iteration stream's index starts with 0."

    batches: list[td.TensorDict] = []
    for idx, batch in enumerate(binary_stream_iter, start=1):
        # Ensure that the input is also exhausted.
        assert idx == binary_stream_iter.idx
        batches.append(batch)

    assert binary_stream_iter.idx == binary_stream.size == len(batches)
