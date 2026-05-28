# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib

import pytest
import torch

from aioway.io._media import TokenizerLoader
from aioway.schemas import DType, IsTokenizedTag


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


def test_tokenize_tags(tokenizer: TokenizerLoader, example_txt: pathlib.Path):
    text = _read_tokenize(tokenizer, example_txt)
    tag = IsTokenizedTag.extract(text)
    assert tag
    assert tag.tokenizer == tokenizer.name
