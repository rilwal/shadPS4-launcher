#!/bin/sh
# Copies the launcher's client/ library here and records the commit it came from.
# Usage: 3rdparty/launcher-client/update.sh /path/to/game-launcher
set -eu
src=${1:?usage: $0 /path/to/game-launcher}
here=$(cd "$(dirname "$0")" && pwd)
test -f "$src/client/include/launcher_client.h" || { echo "$src has no client/ library" >&2; exit 1; }
commit=$(git -C "$src" rev-parse HEAD)
date=$(git -C "$src" log -1 --format=%cs HEAD)
if [ -n "$(git -C "$src" status --porcelain -- client)" ]; then
	echo "warning: $src/client has uncommitted changes" >&2
fi
rm -rf "$here/client"
cp -R "$src/client" "$here/client"
sed -i.bak "s/^Copied from: .*/Copied from: $commit ($date)/" "$here/README.md" && rm -f "$here/README.md.bak"
echo "launcher-client: copied $commit"
