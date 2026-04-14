# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
import torch

from aioway.fn import tdict
from aioway.schemas import Attr, AttrSet, attr, attr_set


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


def test_construction_of_attrset(valid_data: td.TensorDict):
    parsed = tdict(valid_data)
    assert parsed.attrs == attr_set(
        {
            "a": Attr.parse(device="cpu", shape=[11, 2, 3], dtype="int32"),
            "b": Attr.parse(device="cpu", shape=[11, 6], dtype="float32"),
        }
    )


@pytest.fixture
def block(valid_data: td.TensorDict) -> td.TensorDict:
    return valid_data


def test_block_init(block: Chunk):
    _ = block
