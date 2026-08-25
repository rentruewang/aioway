# Copyright (c) AIoWay Authors - All Rights Reserved

apt-get update

command -v pipx || apt-get install pipx -y
command -v ffmpeg || apt-get install ffmpeg -y

command -v pdm || pipx install pdm
