# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import pathlib

import torch
import transformers

__all__ = ["TokenizerLoader"]


@dcls.dataclass(frozen=True)
class TokenizeResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


@dcls.dataclass
class TokenizerLoader:
    name: str
    """
    The tokenizer name to use.
    """

    def __call__(self, fname: str | pathlib.Path, /) -> torch.Tensor:
        fname = pathlib.Path(fname)
        text = fname.read_text()
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        assert isinstance(tokens, torch.Tensor)
        return tokens

    @property
    def tokenizer(self):
        return _tokenizer(self.name)


@functools.cache
def _tokenizer(name: str):
    """
    Load the tokenizer, and cache it in the LRU cache.

    This simply calls `from_pretrained` from huggingface, so name can be a local path.
    """

    return transformers.AutoTokenizer.from_pretrained(name)
