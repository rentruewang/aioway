# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import pathlib

import torch
import transformers

from aioway.schemas import IsTokenizedTag

from ._bases import TorchCompatible

__all__ = ["TokenizerLoader", "TokenizeResult"]


@dcls.dataclass(frozen=True)
class TokenizeResult(TorchCompatible):
    tokenizer: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def to_tensor(self):
        IsTokenizedTag(self.tokenizer).attach(self.input_ids)
        return self.input_ids


@dcls.dataclass
class TokenizerLoader:
    name: str
    """
    The tokenizer name to use.
    """

    def __call__(self, fname: str | pathlib.Path, /) -> TokenizeResult:
        fname = pathlib.Path(fname)
        text = fname.read_text()
        result = self.tokenizer(text, return_tensors="pt", padding=True)
        return TokenizeResult(
            tokenizer=self.name,
            input_ids=result["input_ids"],
            attention_mask=result["attention_mask"],
        )

    @property
    def tokenizer(self):
        return _tokenizer(self.name)


@functools.lru_cache(maxsize=1)
def _tokenizer(name: str):
    """
    Load the tokenizer, and cache it in the LRU cache.

    This simply calls `from_pretrained` from huggingface, so name can be a local path.
    """

    return transformers.AutoTokenizer.from_pretrained(name)
