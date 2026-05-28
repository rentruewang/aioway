# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for `Stream` testing."

import pytest

from aioway.frames import Frame, TdFrame
from aioway.streams import FrameStream, FrameStreamLoader
from tests.fake import chunk_ok, concat_ok, unionable_ok


@pytest.fixture
def block_table(device: str, data_size: int) -> TdFrame:
    block = chunk_ok(size=data_size, device=device)
    return TdFrame(data=block)


@pytest.fixture
def table_stream(block_table: Frame, batch_size: int) -> FrameStream:
    return FrameStream(
        frame=block_table,
        options=FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def concat_frame(device: str, data_size: int) -> TdFrame:
    block = concat_ok(size=data_size, device=device)
    return TdFrame(data=block)


@pytest.fixture
def concat_stream(concat_frame: Frame, batch_size: int) -> FrameStream:
    return FrameStream(
        frame=concat_frame,
        options=FrameStreamLoader(batch_size=batch_size),
    )


@pytest.fixture
def joinable_frame(device: str, data_size: int) -> TdFrame:
    "`Frame` for joining on the RHS."

    block = unionable_ok(size=data_size, device=device)
    return TdFrame(data=block)


@pytest.fixture
def joinable_stream(joinable_frame: Frame, batch_size: int) -> FrameStream:
    "`Stream` for joining on the RHS."
    return FrameStream(
        frame=joinable_frame,
        options=FrameStreamLoader(batch_size=batch_size),
    )
