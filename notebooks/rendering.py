# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import rich

# %%
import torch
from rich import pretty

# %%
from aioway.ir import FuncProc, TorchFuncDag, render_func_proc

# %%
from aioway.t import BoundArgs, fake_mode

pretty.install()

# %% [markdown]
# The following cell will have the following function:
# note that only (x, y) are the inputs.

# %% [markdown]
# step 0: summed = add(x, y)
# step 1: product = mul(summed, x)
# step 2: difference = sub(product, summed)
# step 3: activated = relu(difference)

# %%
tracer = TorchFuncDag()

with fake_mode():
    x, y = torch.zeros(3), torch.zeros(3)

    with tracer.activate():
        summed = torch.add(x, y)
        product = torch.mul(summed, x)
        difference = torch.sub(product, summed)
        activated = torch.relu(difference)

iset = FuncProc.from_instr_list(tracer.thunks)

# For rendering, output doesn't need to be a list.
iset = FuncProc(instrs=iset.instrs, inputs=iset.inputs, outputs=iset.outputs[0])

# %%
render_func_proc(iset, "example_program")

# %%
rich.print(render_func_proc(iset, "example"))

# %%
x, y = iset.inputs.tensors
another = FuncProc(
    instrs=iset.instrs,
    inputs=BoundArgs.from_dict({"mapping": {"a": x, "b": y}}),
    outputs=iset.outputs,
)

# %%
render_func_proc(another, "another_program")

# %%
rich.print(render_func_proc(another, "another"))

# %%
another.__name__ = "new_name"
rich.print(another)
