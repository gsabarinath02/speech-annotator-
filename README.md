# Outcomes Speech Studio

Production speech collection workspace built with Next.js and FastAPI.

## What It Does

- Admins sign in, onboard users, reset user passwords, and manage the scripts users will record.
- Users sign in and record scripts from a clean, low-distraction recording screen.
- Admins can review users and all saved recordings.
- Sessions are tracked server-side, can be revoked on logout, and are revoked when a password is reset.
- Recordings are accepted only as uncompressed 48 kHz WAV files.
- The browser requests raw microphone capture with echo cancellation, noise suppression, and auto gain disabled.
- Saved recordings are written unchanged; the backend records a SHA-256 digest and does not transcode.

## Local Development

Start the API:

```bash
cd backend
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
uvicorn speech_api.main:app --reload --port 8000
```

Start the web app:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

Default local admin:

```text
admin@local.test
Admin@12345
```

## Production Security

Set these values before running with `APP_ENV=production`:

```text
APP_ENV=production
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong password: 12+ chars with upper, lower, number, symbol>
SECRET_KEY=<unique random secret, at least 32 characters>
CORS_ORIGINS=https://your-web-app.example.com
SESSION_TTL_SECONDS=43200
PASSWORD_RESET_TTL_SECONDS=1800
```

In production the API refuses to start if `SECRET_KEY` is missing/weak, the default admin password is still in use, or CORS allows `*`.

## Docker

```bash
docker compose up --build
```

The API runs on `http://localhost:8000` and the web app runs on `http://localhost:3000`.

## Railway Deployment From GitHub

Deploy this repository as two Railway services from the same GitHub repo.

### 1. API Service

Create a Railway service for the backend:

```text
Service name: outcomes-speech-api
Root directory: backend
Config file: backend/railway.toml
```

Set these Railway variables before deploying:

```text
APP_ENV=production
UPLOAD_DIR=/data/uploads
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong password: 12+ chars with upper, lower, number, symbol>
SECRET_KEY=<unique random secret, at least 32 characters>
CORS_ORIGINS=https://your-frontend-service.up.railway.app
SESSION_TTL_SECONDS=43200
PASSWORD_RESET_TTL_SECONDS=1800
```

Attach a Railway volume to the API service and mount it at:

```text
/data/uploads
```

That volume stores the saved WAV files and the app state, so recordings and users survive redeploys.

After the first deploy, generate a public Railway domain for the API service. The API health check is available at:

```text
https://your-api-service.up.railway.app/api/health
```

### 2. Web Service

Create a second Railway service for the frontend:

```text
Service name: outcomes-speech-web
Root directory: frontend
Config file: frontend/railway.toml
```

Set this Railway variable before deploying:

```text
NEXT_PUBLIC_API_BASE_URL=https://your-api-service.up.railway.app
```

Redeploy the web service whenever `NEXT_PUBLIC_API_BASE_URL` changes, because the browser app reads that value during the production build.

### 3. Final Production Checks

- Update the API service `CORS_ORIGINS` to the final web service domain.
- Keep `SECRET_KEY` private and do not reuse the local development secret.
- Use the Railway volume for production recordings; without it, files can be lost on redeploy.
- Push to GitHub after setup. Railway will build from each service root directory and redeploy from the connected branch.

## Verification

```bash
cd backend && pytest
cd frontend && npm test && npm run lint && npm run build
```
