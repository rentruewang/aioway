# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway._schemas import DType
from aioway.codecs import TokenizerLoader


def _tokenizers():
    yield "bert-base-uncased"


@pytest.fixture(params=_tokenizers())
def tokenizer(request: pytest.FixtureRequest):
    return TokenizerLoader(request.param)


def _read_tokenize(tokenizer: TokenizerLoader, txt: pathlib.Path):
    return tokenizer(txt).to_tensor()


def test_tokenize(tokenizer: TokenizerLoader, example_txt: pathlib.Path):
    text = _read_tokenize(tokenizer, example_txt)
    assert isinstance(text, torch.Tensor)
    assert DType.parse(text.dtype).family == "int"
