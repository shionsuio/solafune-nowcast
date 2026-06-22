#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
mkdir -p .cache/matplotlib

export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"

exec .venv/bin/jupyter lab \
    --ip=127.0.0.1 \
    --port=8888 \
    --no-browser \
    --IdentityProvider.token=
