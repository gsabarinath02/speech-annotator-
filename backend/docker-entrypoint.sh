#!/bin/sh
set -eu

mkdir -p "${UPLOAD_DIR:-/data/uploads}"
chown -R appuser:appuser "${UPLOAD_DIR:-/data/uploads}"

exec gosu appuser "$@"
