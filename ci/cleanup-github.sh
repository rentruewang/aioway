# Copyright (c) AIoWay Authors - All Rights Reserved

# Terminate if not in github.
[ "$GITHUB_ACTIONS" = "true" ] || exit 0

echo "Removing files we did not ask for..."

echo "Pruning android..."
rm -rf /usr/local/lib/android

echo "Pruning dotnet..."
rm -rf /usr/share/dotnet

echo "Pruning ghcup..."
rm -rf /usr/local/.ghcup

echo "Pruning docker..."
docker system prune -af --volumes

echo "Investigating how much storage is used in GitHub Actions..."
df -h
