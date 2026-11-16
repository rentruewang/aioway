# Copyright (c) AIoWay Authors - All Rights Reserved

import torch

from aioway.cmds.nets import ImageRegressorEmitter
from aioway.io import float_image_tspec, unbounded_box_tspec
from aioway.t import Shape, all_fake


def test_emit_in_fake_mode(fake_mode_on):
    emitter = ImageRegressorEmitter([5, 6, 7, 8], [1, 2, 3, 4], [1, 2, 3, 4])
    func = emitter(float_image_tspec(3, 28, 28), unbounded_box_tspec(Shape.parse(5)))

    params = list(func.module.parameters())
    assert params and all_fake(params)
    assert func(torch.zeros(7, 3, 28, 28)).shape == (7, 5)
