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

# %%
import rich

from aioway.instrs import Instr


# %%
def find_subclasses(cls: type, seen: set[type]) -> None:
    if cls in seen:
        return
    seen.add(cls)
    for sub in cls.__subclasses__():
        find_subclasses(sub, seen)


# %%
seen: set[type[Instr]] = set()
find_subclasses(Instr, seen)

# %%
classes = sorted(
    seen,
    key=lambda t: [not t.implements_nn(), t.deduction_is_defined(), t.__name__],
)
for c in classes:
    if c.implements_nn():
        rich.print(c, c.deduction_is_defined())
