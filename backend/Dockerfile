FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPLOAD_DIR=/data/uploads

WORKDIR /app

COPY pyproject.toml ./
COPY speech_api ./speech_api
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN apt-get update \
    && apt-get install --no-install-recommends --yes gosu \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir . \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/uploads \
    && chown -R appuser:appuser /data

ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 8000

CMD ["sh", "-c", "uvicorn speech_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
