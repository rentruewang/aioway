# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn

from aioway.might import Might, find_might


def _cases():
    yield nn.Linear, {"in_features": 3, "out_features": 5}
    yield nn.Bilinear, {"in1_features": 3, "in2_features": 3, "out_features": 5}


@pytest.fixture(params=_cases())
def might(request: pytest.FixtureRequest):
    cls, kwargs = request.param
    return find_might(cls, **kwargs)


def test_preview_init(might: Might):
    assert isinstance(might, Might)
    assert repr(might).startswith("might::")


def test_preview_do(might: Might):
    module = might.do()
    assert isinstance(module, nn.Module)
