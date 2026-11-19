# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway.cmds.nets import ImageRegressorEmitter, float_list, float_range_list
from aioway.t import all_fake


def test_emit_in_fake_mode(fake_mode_on):
    emitter = ImageRegressorEmitter([5, 6, 7, 8], [1, 2, 3, 4], [1, 2, 3, 4])
    func = emitter(float_range_list(0, 1, 3, 28, 28), float_list(5))

    params = list(func.module.parameters())
    assert params and all_fake(params)
    assert func(torch.zeros(7, 3, 28, 28)).shape == (7, 5)
