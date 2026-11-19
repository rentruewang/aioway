# Copyright (c) AIoWay Authors - All Rights Reserved

import torch
from torch.utils import data as du

from aioway.cmds.nets import float_list, int_range_item
from aioway.io import DatasetIdxDset, IdxComposedDset, InputTargetDset
from aioway.t import TDictDomain


def _dset(domain, *shape: int) -> DatasetIdxDset:
    return DatasetIdxDset(du.TensorDataset(torch.rand(4, *shape)), domain)


def test_dset_domain():
    assert _dset(float_list(3), 3).__domain__ == float_list(3)


def test_composed_domain():
    dset = IdxComposedDset(x=_dset(float_list(3), 3), y=_dset(float_list(2), 2))
    assert dset.__domain__ == TDictDomain({"x": float_list(3), "y": float_list(2)})


def test_input_target_domain():
    dset = InputTargetDset(_dset(float_list(3), 3), _dset(int_range_item(0, 9)))
    assert dset.__domain__ == TDictDomain(
        {"input": float_list(3), "target": int_range_item(0, 9)}
    )
