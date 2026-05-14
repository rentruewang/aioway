# Copyright (c) AIoWay Authors - All Rights Reserved

import pathlib
import subprocess as sp
import sys


def test_notebook(notebook: pathlib.Path):
    result = sp.run(
        [sys.executable, str(notebook)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"process failed for {notebook}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
