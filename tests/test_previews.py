# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from torch import nn
from aioway.previews import Linear, Bilinear
from aioway.previews import Preview
from aioway.previews import find_preview


def _cases():
    yield nn.Linear, {"in_features": 3, "out_features": 5}
    yield nn.Bilinear, {"in1_features": 3, "in2_features": 3, "out_features": 5}


@pytest.fixture(params=_cases())
def preview(request: pytest.FixtureRequest):
    cls, kwargs = request.param
    return find_preview(cls, **kwargs)


def test_preview_init(preview: Preview):
    assert isinstance(preview, Preview)
    assert repr(preview)


def test_preview_do(preview: Preview):
    module = preview.do()
    assert isinstance(module, nn.Module)
