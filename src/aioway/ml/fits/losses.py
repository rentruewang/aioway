# Copyright (c) AIoWay Authors - All Rights Reserved

import enum

from aioway.cmds.nets import float_list
from aioway.t import SizedMixin


class FitType(enum.StrEnum):
    "The type of fitting to use."

    REGRESSION = "regression"
    "Where the output is a floating vector."

    CLASSIFICATION = "classification"
    "Where an output can match 1 single class."

    MULTI_CLASS = "multi-class"
    "Where an output can match multiple classes."

    def target_to_input_domain(self, target: SizedMixin, /) -> SizedMixin:
        match self:
            case self.REGRESSION:
                func = self._reg_input

            case self.CLASSIFICATION:
                func = self._clf_input

            case self.MULTI_CLASS:
                func = self._multi_clf_input

        return func(target)

    def _reg_input(self, target: SizedMixin) -> SizedMixin:
        return target

    def _clf_input(self, target: SizedMixin) -> SizedMixin:
        "Float logits, one per class, any number of classes."

        return float_list(*target.shape, -1)

    def _multi_clf_input(self, target: SizedMixin) -> SizedMixin:
        "Float logits, one per class, any number of classes."

        return float_list(*target.shape, -1)
