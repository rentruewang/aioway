# Copyright (c) AIoWay Authors - All Rights Reserved

"The shared utilities for `Stream` testing."

import pytest

from aioway.dsets import TdictFrame, TensorDictFrame
from aioway.relalg import LoaderIter, LoaderOpt
from tests.mock import chunk_ok, concat_ok, unionable_ok


@pytest.fixture
def block_table(device: str, data_size: int) -> TensorDictFrame:
    block = chunk_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def table_stream(block_table: TdictFrame, batch_size: int) -> LoaderIter:
    return block_table(LoaderOpt(batch_size=batch_size))


@pytest.fixture
def concat_frame(device: str, data_size: int) -> TensorDictFrame:
    block = concat_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def concat_stream(concat_frame: TdictFrame, batch_size: int) -> LoaderIter:
    return concat_frame(LoaderOpt(batch_size=batch_size))


@pytest.fixture
def joinable_frame(device: str, data_size: int) -> TensorDictFrame:
    "`Frame` for joining on the RHS."

    block = unionable_ok(size=data_size, device=device)
    return TensorDictFrame(data=block)


@pytest.fixture
def joinable_stream(joinable_frame: TdictFrame, batch_size: int) -> LoaderIter:
    "`Stream` for joining on the RHS."
    return joinable_frame(LoaderOpt(batch_size=batch_size))
