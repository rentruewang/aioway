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

To setup the environment (including development dependencies) with `nox`, run

```bash
nox -s setup
```

To run testing, do:

```bash
nox -s test
```

To format the codebase, run:

```bash
nox -s format
```

To just check the formatting (no modification), run:

```bash
nox -s format_check
```

To run type checking with `mypy`, run:

```bash
nox -s type
```

`format`, `type`, `test` are checked in github actions, and they are required for merging.

Run `nox -l` for all options defined in the project.

## Recommended development style

### Python code style

Please write code matching the style of the surrounding code.

Otherwise, follow the following style guide that I personally use (by me): [link](https://github.com/rentruewang/mind/blob/main/py/CONTRIBUTING.md).

### Documentation

The documentation string style follows the Google style format specified [here](https://mkdocstrings.github.io/griffe/docstrings/#google-style).

### Commit message

Commit message should follow the format (applied during squash and merge):

```
{emoji} Message. [([fix] #{issue})]

Detailed explanation.
```

Where `[]` denotes optional in the above commit message. `{emoji}` should be a relevant emoji, and `#{issue}` should be the relevant issue number.
