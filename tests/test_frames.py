# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
from numpy import random

from aioway._frames import TdictFrame
from aioway.io import TensorDictFrame, TensorDictListFrame
from aioway.relalg import FrameStream, FrameStreamLoader
from tests.fake import chunk_ok


def block_table(device: str, batch_size: int, data_size: int):
    block = chunk_ok(size=data_size, device=device)
    return TensorDictFrame(block)


def list_table(device: str, batch_size: int, data_size: int):
    return TensorDictListFrame(
        [
            chunk_ok(size=batch_size, device=device)
            for _ in range(0, data_size, batch_size)
        ]
    )


@pytest.fixture(params=[block_table, list_table])
def frame(
    request: pytest.FixtureRequest, device: str, batch_size: int, data_size: int
) -> TdictFrame:
    return request.param(device=device, batch_size=batch_size, data_size=data_size)


@pytest.fixture
def table_stream(frame: TdictFrame, batch_size: int):
    return FrameStream(
        frame,
        FrameStreamLoader(batch_size=batch_size),
    )


def test_table_not_empty(frame: TdictFrame):
    assert frame
    assert len(frame)


def test_table_idx_arr(frame: TdictFrame):
    idx = random.randint(low=-len(frame), high=len(frame), size=[len(frame)])

    assert np.all(-len(frame) <= idx)
    assert np.all(idx < len(frame))
    assert idx.shape == (len(frame),)

    out = frame.__getitems__(idx.tolist())
    assert isinstance(out, td.TensorDict)
    assert len(out) == len(idx)


def test_table_out_of_bounds(frame: TdictFrame):
    with pytest.raises(IndexError):
        _ = frame.__getitems__([-2 * len(frame)])


def test_column_attr(frame: TdictFrame) -> None:
    attrs = frame.attrs
    first_key = list(attrs.keys())[0]

    assert frame.column(first_key).attr == attrs[first_key]
    assert frame.select(first_key).attrs == {first_key: attrs[first_key]}


def test_select_attr(frame: TdictFrame) -> None:
    attrs = frame.attrs
    k_0, k_1 = list(attrs.keys())[:2]

    selected = {k_0: attrs[k_0], k_1: attrs[k_1]}
    assert frame.select(k_0, k_1).attrs == selected
