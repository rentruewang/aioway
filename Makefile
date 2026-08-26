# Copyright (c) AIoWay Authors - All Rights Reserved

BASH := python3 ci/launch.py bash
SHELL := $(if $(BASH),$(BASH),bash)

OS := $(shell uname -s)
PYTEST_FLAGS := 
PYTHON_VERSION :=
CHECK :=
CHECK_FLAG := $(if $(CHECK),--check,)
SUDO := sudo -E


setup: cleanup deps

cleanup:
	@$(SUDO) $(SHELL) ci/cleanup-github.sh

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	@$(SUDO) $(SHELL) ci/install-linux.sh
else ifeq ($(OS),Darwin)
	@$(SHELL) ci/install-mac.sh
else
	@echo "Unsupported OS: $(OS)"
	@exit 1
endif

publish:
	@$(SHELL) ci/pdm.sh publish

build:
	@$(SHELL) ci/pdm.sh build

install:
	@$(SHELL) ci/pdm.sh install "-G:all"

sync:
	@$(SHELL) ci/pdm.sh sync "-G:all"

pytest:
	@$(SHELL) ci/pdm.sh run pytest $(PYTEST_FLAGS)

autoflake:
	@$(SHELL) ci/pdm.sh run autoflake . $(CHECK_FLAG)

black:
	@$(SHELL) ci/pdm.sh run black . $(CHECK_FLAG)

isort:
	@$(SHELL) ci/pdm.sh run isort . $(CHECK_FLAG)

mypy:
	@$(SHELL) ci/pdm.sh run mypy --install-types --non-interactive src


sphinx:
	@$(SHELL) ci/pdm.sh run make -C docs html

docs: sphinx
	@$(SHELL) ci/docs.sh
