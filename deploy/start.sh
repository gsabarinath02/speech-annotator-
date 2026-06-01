#!/bin/sh
set -eu

export PORT="${PORT:-8080}"
export UPLOAD_DIR="${UPLOAD_DIR:-/data/uploads}"

mkdir -p "$UPLOAD_DIR"
chown -R appuser:appuser "$UPLOAD_DIR"

envsubst '${PORT}' < /etc/nginx/templates/speech-studio.conf.template > /etc/nginx/conf.d/default.conf
rm -f /etc/nginx/sites-enabled/default

exec supervisord -c /etc/supervisor/conf.d/speech-studio.conf
