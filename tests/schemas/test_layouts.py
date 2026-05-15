# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway.schemas import Layout, LayoutLike


def _layouts():
    yield torch.strided
    yield torch.sparse_coo

    yield "strided"
    yield "sparse_coo"


@pytest.fixture(params=_layouts())
def layout_like(request: pytest.FixtureRequest):
    return request.param


def test_layout_parse(layout_like: LayoutLike):
    assert Layout.parse(layout_like)


def test_layout_eq(layout_like: LayoutLike):
    assert Layout.parse(layout_like) == layout_like
