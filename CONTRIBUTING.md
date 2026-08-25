# Contributing

> We do not welcome purely vibe coded PRs.
> PRs authors should demonstrate deep understanding of the code they wrote before the PR can be merged.

## Development installation

First, clone and navigate into the project:

```bash
git clone https://github.com/rentruewang/aioway
cd aioway/
```

Alternatively, use ssh:
```bash
git clone git@github.com:rentruewang/aioway
cd aioway/
```

I'm using [nox](https://nox.thea.codes/) for build management,
and [pdm](https://pdm-project.org/) in this project for dependency management.

To setup the environment (including development dependencies) with `pdm`, run

```bash
pdm install -G:all
```

Then activate with

```bash
eval $(pdm venv activate)
```

or simpler, if you want to use our makefile:

```bash
make setup install
```

To run testing, do:

```bash
make pytest
```

To format the codebase, run:

```bash
make autoflake isort black
```

To just check the formatting (no modification), run:

```bash
make autoflake isort black CHECK=1
```

To run type checking with `mypy`, run:

```bash
make mypy
```

These make commands are executed in github actions, see `.github/workflows` for details.

## Recommended development style

### Python code style

Please write code matching the style of the surrounding code.

Otherwise, follow the following style guide that I personally use (by me): [link](https://github.com/rentruewang/mind/blob/main/CONTRIBUTING.md).

### Documentation

The documentation string style follows the Google style format specified [here](https://mkdocstrings.github.io/griffe/docstrings/#google-style).

### Commit message

Commit message should follow the format (applied during squash and merge):

```
{emoji} Message. [([fix] #{issue})]

Detailed explanation.
```

Where `[]` denotes optional in the above commit message. `{emoji}` should be a relevant emoji, and `#{issue}` should be the relevant issue number.
