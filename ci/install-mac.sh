# Copyright (c) AIoWay Authors - All Rights Reserved

command -v brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

command -v pipx || brew install pipx -y
command -v ffmpeg || brew install ffmpeg -y
command -v timeout || brew install coreutils -y

command -v pdm || pipx install pdm
