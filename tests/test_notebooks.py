# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import subprocess as sp
import sys, papermill as pm


def test_notebook(notebook: pathlib.Path):
    output_notebook = notebook.with_suffix(".out.ipynb")

    # Execute the notebook (raises pm.PapermillExecutionError if a cell fails)
    pm.execute_notebook(input_path=notebook, output_path=str(output_notebook))


def test_notebook_is_clean(notebook: pathlib.Path):
    if result := sp.call(["nox", "-rs", "nb_check", "--", str(notebook)]):
        raise AssertionError(f"{notebook} is not clean.")
