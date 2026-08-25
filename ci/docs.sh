# Copyright (c) AIoWay Authors - All Rights Reserved

mkdir -p "public"

if [ ! -f "public/.gitignore" ]; then
    printf '%s\n' '*' > "public/.gitignore"
fi

rsync -a "docs/" "public/"
rsync -a "docs/build/html/" "public/api/"
