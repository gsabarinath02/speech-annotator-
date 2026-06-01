#!/bin/sh
set -eu

export PORT="${PORT:-8080}"
export WEB_INTERNAL_PORT="${WEB_INTERNAL_PORT:-3001}"
export UPLOAD_DIR="${UPLOAD_DIR:-/data/uploads}"

mkdir -p "$UPLOAD_DIR"
chown -R appuser:appuser "$UPLOAD_DIR"

envsubst '${PORT} ${WEB_INTERNAL_PORT}' < /etc/nginx/templates/speech-studio.conf.template > /etc/nginx/conf.d/default.conf
rm -f /etc/nginx/sites-enabled/default

nginx -t

exec supervisord -c /etc/supervisor/conf.d/speech-studio.conf
