# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import importlib
from importlib import util as impu


def test_notebook(notebook: pathlib.Path):
    module_name = notebook.with_suffix("").name

    # Perform an import on the notebook (which are pypercent files), and execute it.
    spec = impu.spec_from_file_location(f"notebooks.{module_name}", notebook)
    assert spec
    assert spec.loader
    module = impu.module_from_spec(spec)
    spec.loader.exec_module(module)
