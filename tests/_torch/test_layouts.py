# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch

from aioway._torch import Layout, LayoutLike


def _layouts():
    yield torch.strided
    yield torch.sparse_bsc
    yield torch.sparse_bsr
    yield torch.sparse_coo
    yield torch.sparse_csc
    yield torch.sparse_csr

    yield "strided"
    yield "sparse_bsc"
    yield "sparse_bsr"
    yield "sparse_coo"
    yield "sparse_csc"
    yield "sparse_csr"


@pytest.fixture(params=_layouts())
def layout_like(request: pytest.FixtureRequest):
    return request.param


def test_layout_parse(layout_like: LayoutLike):
    assert Layout.parse(layout_like)


def test_layout_eq(layout_like: LayoutLike):
    assert Layout.parse(layout_like) == layout_like
