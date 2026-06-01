FROM node:22-bookworm-slim AS frontend-deps

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

ARG NEXT_PUBLIC_API_BASE_URL=""
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=frontend-deps /app/frontend/node_modules ./node_modules
COPY frontend ./

RUN npm run build

FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPLOAD_DIR=/data/uploads \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        gettext-base \
        gosu \
        nginx \
        python3 \
        python3-venv \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/speech_api ./backend/speech_api

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir ./backend \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/uploads /var/log/supervisor \
    && chown -R appuser:appuser /data /app

COPY --from=frontend-builder /app/frontend/.next/standalone ./frontend
COPY --from=frontend-builder /app/frontend/.next/static ./frontend/.next/static
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY deploy/nginx.conf.template /etc/nginx/templates/speech-studio.conf.template
COPY deploy/supervisord.conf /etc/supervisor/conf.d/speech-studio.conf
COPY deploy/start.sh /usr/local/bin/start-speech-studio

RUN chmod +x /usr/local/bin/start-speech-studio

EXPOSE 8080

ENTRYPOINT ["start-speech-studio"]
