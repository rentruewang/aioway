# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
from numpy import random

from aioway.dsets import Frame


def test_table_not_empty(frame: Frame):
    assert frame
    assert len(frame)


def test_table_idx_arr(frame: Frame):
    idx = random.randint(low=-len(frame), high=len(frame), size=[len(frame)])

    assert np.all(-len(frame) <= idx)
    assert np.all(idx < len(frame))
    assert idx.shape == (len(frame),)

    out = frame[idx.tolist()]
    assert isinstance(out, td.TensorDict)
    assert len(out) == len(idx)


def test_table_idx_slice(frame: Frame):
    lf = len(frame)
    out = frame[-lf:lf]
    assert isinstance(out, td.TensorDict)
    assert len(out) == len([*range(lf)[-lf:lf]])


def test_table_out_of_bounds(frame: Frame):
    with pytest.raises(IndexError):
        _ = frame[[-2 * len(frame)]]
