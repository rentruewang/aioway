# Copyright (c) AIoWay Authors - All Rights Reserved

OS := $(shell uname -s)

setup: cleanup deps


cleanup:
	@python launch.py bash cleanup-github.sh

deps:
	@echo "Installing dependencies for $(OS)"

ifeq ($(OS),Linux)
	@python launch.py bash install-linux.sh
else ifeq ($(OS),Darwin)
	@python launch.py bash install-mac.sh
else
	@echo "Unsupported OS: $(OS)"
	@exit 1
endif

publish:
	@python launch.py bash pdm.sh publish

build:
	@python launch.py bash pdm.sh build

install:
	@python launch.py bash pdm.sh install "-G:all"

sync:
	@python launch.py bash pdm.sh sync "-G:all"
