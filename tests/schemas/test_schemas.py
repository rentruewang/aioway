# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td
import torch

from aioway.schemas import Attr, IsImageTag, Schema, SchemaDict


@pytest.fixture
def image():
    return torch.randint(0, 255, [3, 100, 100]).to(torch.uint8)


@pytest.fixture
def tagged_image(image: torch.Tensor):
    IsImageTag().attach(image)
    return image


@pytest.fixture
def tagged_image_tdict(tagged_image: torch.Tensor):
    assert IsImageTag.extract(tagged_image) is not None
    return td.TensorDict({"image": tagged_image})


def test_schema_of_image(tagged_image: torch.Tensor):
    schema = Schema.from_tensor(tagged_image)
    assert schema.attr == Attr.parse(tagged_image)
    assert schema.tags == {IsImageTag.TAG: IsImageTag()}


def test_schema_of_image_dict(
    tagged_image_tdict: td.TensorDict, tagged_image: torch.Tensor
):
    schema = SchemaDict.from_tensor_mapping(tagged_image_tdict)
    assert schema["image"].attr == Attr.parse(tagged_image)
    assert schema["image"].tags == {IsImageTag.TAG: IsImageTag()}
