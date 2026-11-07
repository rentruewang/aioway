# Py Trees

This is the compatible package based on `torch`'s `pytree` package,
which is available in `from torch.utils import _pytree`.

# Compatibility

It attempts to reuse the more performant `optree` implementation,
available at `from torch.utils import _cxx_pytree`,
and fills in the operation that it does not implement with the normal `pytree` version,
mostly around exporting paths.

Since `optree` version uses a different `TreeSpec`,
extra caution is taken to ensure the return spec is the same,
at the small cost of running both `optree` and `pytree` versions for certain functions.

See `compat.py` for details.
