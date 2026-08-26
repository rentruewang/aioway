# Copyright (c) AIoWay Authors - All Rights Reserved

SH := python3 ci/launch.py bash

OS := $(shell uname -s)
PYTEST_FLAGS := 
PYTHON_VERSION :=
CHECK :=
CHECK_FLAG := $(if $(CHECK),--check,)
SUDO := sudo -E


setup: cleanup deps

cleanup:
	@$(SUDO) $(SH) ci/cleanup-github.sh

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	@$(SUDO) $(SH) ci/install-linux.sh
else ifeq ($(OS),Darwin)
	@$(SH) ci/install-mac.sh
else
	@echo "Unsupported OS: $(OS)"
	@exit 1
endif

publish:
	@$(SH) ci/pdm.sh publish

build:
	@$(SH) ci/pdm.sh build

install:
	@$(SH) ci/pdm.sh install "-G:all"

sync:
	@$(SH) ci/pdm.sh sync "-G:all"

pytest:
	@$(SH) ci/pdm.sh run pytest $(PYTEST_FLAGS)

autoflake:
	@$(SH) ci/pdm.sh run autoflake . $(CHECK_FLAG)

black:
	@$(SH) ci/pdm.sh run black . $(CHECK_FLAG)

isort:
	@$(SH) ci/pdm.sh run isort . $(CHECK_FLAG)

mypy:
	@$(SH) ci/pdm.sh run mypy --install-types --non-interactive src


sphinx:
	@$(SH) ci/pdm.sh run make -C docs html

docs: sphinx
	@$(SH) ci/docs.sh
