# Copyright (c) AIoWay Authors - All Rights Reserved

BASH := python3 ci/launch.py bash
LAUNCH := $(if $(BASH),$(BASH),bash)

OS := $(shell uname -s)
PYTEST_FLAGS := 
PYTHON_VERSION :=
CHECK :=
CHECK_FLAG := $(if $(CHECK),--check,)
SUDO := sudo -E


setup: cleanup deps

cleanup:
	@$(SUDO) $(LAUNCH) ci/cleanup-github.sh

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	@$(SUDO) $(LAUNCH) ci/install-linux.sh
else ifeq ($(OS),Darwin)
	@$(LAUNCH) ci/install-mac.sh
else
	@echo "Unsupported OS: $(OS)"
	@exit 1
endif

publish:
	@$(LAUNCH) ci/pdm.sh publish

build:
	@$(LAUNCH) ci/pdm.sh build

install:
	@$(LAUNCH) ci/pdm.sh install "-G:all"

sync:
	@$(LAUNCH) ci/pdm.sh sync "-G:all"

pytest:
	@$(LAUNCH) ci/pdm.sh run pytest $(PYTEST_FLAGS)

autoflake:
	@$(LAUNCH) ci/pdm.sh run autoflake . $(CHECK_FLAG)

black:
	@$(LAUNCH) ci/pdm.sh run black . $(CHECK_FLAG)

isort:
	@$(LAUNCH) ci/pdm.sh run isort . $(CHECK_FLAG)

mypy:
	@$(LAUNCH) ci/pdm.sh run mypy --install-types --non-interactive src


sphinx:
	@$(LAUNCH) ci/pdm.sh run make -C docs html

docs: sphinx
	@$(LAUNCH) ci/docs.sh
