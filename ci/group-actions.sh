# Copyright (c) AIoWay Authors - All Rights Reserved

echo "::group::$*"
trap 'echo "::endgroup::"' EXIT
"$@" 2>&1
