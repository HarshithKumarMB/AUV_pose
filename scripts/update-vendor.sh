#!/usr/bin/env bash
# Verify vendor/holoocean against upstream.
#
# The vendored client is a copy, not a submodule (see vendor/holoocean/VENDOR.md for
# why). This reproduces the guarantee a submodule pointer would have given: that the
# vendored source is byte-identical to a specific upstream tag.
#
#   ./scripts/update-vendor.sh          # check against the pinned tag
#   ./scripts/update-vendor.sh v2.4.0   # check against a different tag
#
# Exits non-zero if the trees differ.

set -euo pipefail

URL="git@github.com:byu-holoocean/HoloOcean.git"
TAG="${1:-v2.3.0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor="$repo_root/vendor/holoocean"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching $URL at $TAG ..."
# Upstream carries ~1.2 GB of Unreal engine content. blob:none + sparse keeps this to
# the client directory only.
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" \
    --filter=blob:none --sparse "$URL" "$tmp/up"
git -C "$tmp/up" sparse-checkout set client

commit="$(git -C "$tmp/up" rev-parse HEAD)"
echo "Upstream $TAG = $commit"
echo

status=0

# src/ is the package -- this is the comparison that matters.
if diff -r -x '__pycache__' "$tmp/up/client/src" "$vendor/client/src"; then
    echo "OK  client/src matches $TAG"
else
    echo "DIFF  client/src differs from $TAG"
    status=1
fi

for f in setup.py README.md; do
    if diff -q "$tmp/up/client/$f" "$vendor/client/$f" >/dev/null; then
        echo "OK  client/$f matches $TAG"
    else
        echo "DIFF  client/$f differs from $TAG"
        status=1
    fi
done

echo
if [ "$status" -eq 0 ]; then
    echo "Vendored tree is an unmodified copy of $TAG ($commit)."
else
    echo "Vendored tree has diverged. Re-copy client/src, client/setup.py and"
    echo "client/README.md from $TAG, then update the table in vendor/holoocean/VENDOR.md."
fi

exit "$status"
