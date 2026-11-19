# Copyright (c) AIoWay Authors - All Rights Reserved

from aioway.cmds.nets import float_list, int_range_item
from aioway.ml.fits.losses import FitType


def test_regression_keeps_target():
    target = float_list(3)
    assert FitType.REGRESSION.target_to_input_domain(target) == target


def test_classification_gives_logits():
    target = int_range_item(0, 9)
    assert FitType.CLASSIFICATION.target_to_input_domain(target) == float_list(-1)
    assert FitType.MULTI_CLASS.target_to_input_domain(target) == float_list(-1)
