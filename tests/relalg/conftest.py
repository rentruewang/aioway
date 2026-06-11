# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for `Stream` testing."

import pytest

from aioway.io import FrameHop, FrameHopLoader, TdictFrame, TensorDictFrame
from tests.fake import chunk_ok, concat_ok, unionable_ok


@pytest.fixture
def block_table(device: str, data_size: int) -> TensorDictFrame:
    block = chunk_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def table_stream(block_table: TdictFrame, batch_size: int) -> FrameHop:
    return FrameHop(
        frame=block_table,
        options=FrameHopLoader(batch_size=batch_size),
    )


@pytest.fixture
def concat_frame(device: str, data_size: int) -> TensorDictFrame:
    block = concat_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def concat_stream(concat_frame: TdictFrame, batch_size: int) -> FrameHop:
    return FrameHop(
        frame=concat_frame,
        options=FrameHopLoader(batch_size=batch_size),
    )


@pytest.fixture
def joinable_frame(device: str, data_size: int) -> TensorDictFrame:
    "`Frame` for joining on the RHS."

    block = unionable_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def joinable_stream(joinable_frame: TdictFrame, batch_size: int) -> FrameHop:
    "`Stream` for joining on the RHS."
    return FrameHop(
        frame=joinable_frame,
        options=FrameHopLoader(batch_size=batch_size),
    )
