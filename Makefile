# Copyright (c) AIoWay Authors - All Rights Reserved

LAUNCH := python3 ci/launch.py bash
OS := $(shell uname -s)
PYTEST_ARGS := 
PYTHON_VERSION :=

setup: cleanup deps

cleanup:
	@$(LAUNCH) ci/cleanup-github.sh

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	@$(LAUNCH) ci/install-linux.sh
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
	@$(LAUNCH) ci/pdm.sh run pytest $(PYTEST_ARGS)

sphinx:
	@$(LAUNCH) ci/pdm.sh run make -C docs html

docs: sphinx
	@$(LAUNCH) ci/docs.sh
