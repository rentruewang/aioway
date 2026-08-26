# Copyright (c) AIoWay Authors - All Rights Reserved


OS := $(shell uname -s)
PYTEST_FLAGS := 
PYTHON_VERSION :=
CHECK :=
CHECK_FLAG := $(if $(CHECK),--check,)
SUDO := sudo -E

# Running in GitHub Actions
ifeq ($(GITHUB_ACTIONS),true)
    SH := python3 ci/group-actions.py
	CLEANUP := $(SUDO) $(SH) bash ci/cleanup-github.sh
# Running locally
else
    SH := bash ci/echo-run.sh
	CLEANUP :=
endif


setup: cleanup deps

cleanup:
	$(CLEANUP)

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	$(SUDO) $(SH) bash ci/install-linux.sh
else ifeq ($(OS),Darwin)
	@$(SH) bash ci/install-mac.sh
else
	@echo "Unsupported OS: $(OS)"
	@exit 1
endif

publish:
	@$(SH) pdm publish

build:
	@$(SH) pdm build

install:
	@$(SH) pdm install "-G:all"

sync:
	@$(SH) pdm sync "-G:all"

pytest:
	@$(SH) pdm run pytest $(PYTEST_FLAGS)

autoflake:
	@$(SH) pdm run autoflake . $(CHECK_FLAG)

black:
	@$(SH) pdm run black . $(CHECK_FLAG)

isort:
	@$(SH) pdm run isort . $(CHECK_FLAG)

mypy:
	@$(SH) pdm run mypy --install-types --non-interactive src


sphinx:
	@$(SH) pdm run make -C docs html

docs: sphinx
	@$(SH) bash ci/docs.sh
