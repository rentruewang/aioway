# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
import torch

from aioway.schemas import Attr, AttrSet, attr


@pytest.fixture
def schema() -> AttrSet:
    return AttrSet.from_values(
        a=attr(
            {
                "device": "cpu",
                "dtype": "int32",
                "shape": [-1, 2, 3],
            },
        ),
        b=attr(
            {
                "device": "cpu",
                "dtype": "float32",
                "shape": [-1, 6],
            },
        ),
    )


@pytest.fixture
def valid_data() -> td.TensorDict:
    result = td.TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    )
    result.auto_batch_size_()
    return result


def _invalid_data():
    # Invalid shape
    yield td.TensorDict(
        {
            "a": torch.randn(11, 2, 3, 4).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()

    # Invalid dtype
    yield td.TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int64),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()


@pytest.fixture(params=_invalid_data())
def invalid_data(request: pytest.FixtureRequest) -> td.TensorDict:
    return request.param


def test_attrset_getitem(schema: AttrSet):
    assert isinstance(schema["a"], Attr)
    assert isinstance(schema[["a", "b"]], AttrSet)
    assert schema == schema[["a", "b"]]
    assert isinstance(schema[[-1, 2, 3]], AttrSet)
    assert isinstance(schema[np.array([-1, 2, 3])], AttrSet)


@pytest.fixture
def block(valid_data: td.TensorDict) -> td.TensorDict:
    return valid_data


def test_block_init(block: td.TensorDict):
    _ = block
