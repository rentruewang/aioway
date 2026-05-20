# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import subprocess as sp

import papermill as pm


def test_notebook(notebook: pathlib.Path):
    output_notebook = notebook.with_suffix(".out.ipynb")

    # Execute the notebook (raises pm.PapermillExecutionError if a cell fails)
    pm.execute_notebook(input_path=notebook, output_path=str(output_notebook))
