# `NnInit`s

This module corresponds to `nn.Module.__init__` signatures.
For example, `Linear` corresponds to `nn.Linear`,
and can be (and should be) used to initalize the `nn.Module`,
as it provies additional checks as well as proper `module_init` wrapping,
so that our `NnInitMode` works (allowing for rewrite / logging of `nn.Module.__init__`).
