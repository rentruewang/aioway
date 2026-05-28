# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
from numpy import random

from aioway._frames import TdictFrame


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
