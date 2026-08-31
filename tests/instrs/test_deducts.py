# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest


@pytest.fixture
def use_new_deductors():
    with new_deductor_registry():
        yield
