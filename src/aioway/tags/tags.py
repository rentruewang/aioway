# Copyright (c) AIoWay Authors - All Rights Reserved

"Extra information about the tensors."

import abc
import dataclasses as dcls
import re
import typing

import torch

from aioway._common import dcls_frozen_slots_no_eq
from aioway.fake import is_fake_tensor

__all__ = ["Tag", "extract_tags"]

_TAG_NAME = re.compile(r"^__aioway_[a-zA-Z0-9_]+__$")


@dcls_frozen_slots_no_eq
class Tag(abc.ABC):
    """
    The base class for tags. A tag describes a `torch.Tensor`,
    and is piggybacked onto the tensor to pass around `torch` APIs.

    Each tag type (subclass) defines a tag name `cls.TAG`,
    corresponding to the attribute name it is set on the tensor.

    This means each tensor would have 1 tag of the same type,
    multiple tags would have multiple types. Therefore, an instance of tag
    is expected to fully describe a tensor in that tag's expertise.
    For example, a tag about batch dimensino should describe all dimensions at once,
    because that tag type can only have a singleton on each tensor.
    """

    TAG: typing.ClassVar[str]
    """
    The name of the tag. Must be of the format `__aioway_*__`.
    """

    tensor: torch.Tensor
    "The tensor that is being piggybacked."

    def __init_subclass__(cls) -> None:
        if not _TAG_NAME.fullmatch(cls.TAG):
            raise ValueError(
                f"Tag name should be of the format `__aioway_*__`. Got {cls.TAG}."
            )

    def __post_init__(self) -> None:
        # Set the tag onto the tensor.
        setattr(self.tensor, self.TAG, self)

    @typing.override
    def __eq__(self, other: object):
        "Comparing 2 tags compare all fields that are not the `.tensor`."

        if type(self) == type(other):
            assert isinstance(other, Tag)
            return self._cmp_dict() == other._cmp_dict()

        return NotImplemented

    def _cmp_dict(self):
        fields = {f.name for f in dcls.fields(self)}
        base_tag_fields = {f.name for f in dcls.fields(Tag)}
        to_compare = fields - base_tag_fields
        return {key: getattr(self, key) for key in to_compare}

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    @property
    def is_fake(self) -> bool:
        return is_fake_tensor(self.tensor)

    def attach(self, other: torch.Tensor, /) -> typing.Self:
        """
        Tag the instance on another tensor. These 2 tags would be `==` to each other.
        """

        # In case the same tensor is tagged again.
        if self.tensor is other:
            return self

        # Replace the `tensor` attribute,
        # and call `__post_init__` which handles attribute setting.
        copied = dcls.replace(self, tensor=other)
        copied.__post_init__()
        assert copied == self.extract(other)
        return copied

    @classmethod
    def extract(cls, tensor: torch.Tensor) -> typing.Self | None:
        """
        Get the tag currently stored on the `tensor`.
        """

        if (tag := getattr(tensor, cls.TAG, None)) is None:
            return None

        if not isinstance(tag, cls):
            raise TypeError(f"The tag {tag} is of type {type(tag)}, expected {cls}.")

        return tag


def extract_tags(tensor: torch.Tensor, /) -> dict[str, Tag]:
    """
    Extract all the tags on the `torch.Tensor`.

    This assumes that no one else uses the `__aioway_*__` namespace. Hopefully.
    """

    tag_names = [attr_name for attr_name in dir(tensor) if _TAG_NAME.match(attr_name)]
    return {name: _get_tag(tensor, name) for name in tag_names}


def _get_tag(tensor: torch.Tensor, tag_name: str, /) -> Tag:
    if isinstance(attr := getattr(tensor, tag_name, None), Tag):
        return attr

    raise AttributeError(
        f"The attribute '{tag_name}' is either not found or not a tag."
    )
