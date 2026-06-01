# Outcomes Speech Studio

Production speech collection workspace built with Next.js and FastAPI.

## What It Does

- Admins sign in, onboard users, reset user passwords, and manage the scripts users will record.
- Users sign in and record scripts from a clean, low-distraction recording screen.
- Admins can review users and all saved recordings.
- Sessions are tracked server-side, can be revoked on logout, and are revoked when a password is reset.
- Users, sessions, reset tokens, scripts, recording metadata, and best-take choices are stored in PostgreSQL.
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
DATABASE_URL=postgresql://user:password@host:5432/database
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong password: 12+ chars with upper, lower, number, symbol>
SECRET_KEY=<unique random secret, at least 32 characters>
CORS_ORIGINS=https://your-web-app.example.com
SESSION_TTL_SECONDS=43200
PASSWORD_RESET_TTL_SECONDS=1800
```

In production the API refuses to start if `DATABASE_URL` is missing, `SECRET_KEY` is missing/weak, the default admin password is still in use, or CORS allows `*`.

## Docker

```bash
docker compose up --build
```

Docker Compose starts PostgreSQL, the API, and the web app. The API runs on `http://localhost:8000` and the web app runs on `http://localhost:3000`.

To run the same single-container image used by Railway:

```bash
docker build -t outcomes-speech-studio:local .
docker run --rm -p 8080:8080 \
  -e APP_ENV=production \
  -e DATABASE_URL='postgresql://speech:speechpass@host.docker.internal:5432/speech_studio' \
  -e UPLOAD_DIR=/data/uploads \
  -e ADMIN_EMAIL=admin@example.com \
  -e ADMIN_PASSWORD='Admin@12345!' \
  -e SECRET_KEY='replace-with-a-long-random-production-secret' \
  -e CORS_ORIGINS=http://localhost:8080 \
  outcomes-speech-studio:local
```

Open `http://localhost:8080`.

## Railway Deployment From GitHub

The recommended Railway setup is one app container from the repository root plus one Railway PostgreSQL service. Nginx serves one public URL, proxies `/api` to FastAPI, and sends all other traffic to Next.js.

Create one Railway service from the GitHub repo:

```text
Root directory: leave empty / repo root
Config file: railway.toml
Dockerfile: Dockerfile
```

Add a PostgreSQL service in the same Railway project. Then set these variables on the app service before deploying:

```text
APP_ENV=production
DATABASE_URL=${{Postgres.DATABASE_URL}}
UPLOAD_DIR=/data/uploads
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=<strong password: 12+ chars with upper, lower, number, symbol>
SECRET_KEY=<unique random secret, at least 32 characters>
CORS_ORIGINS=https://your-service.up.railway.app
SESSION_TTL_SECONDS=43200
PASSWORD_RESET_TTL_SECONDS=1800
```

Attach a Railway volume to the service and mount it at:

```text
/data/uploads
```

That volume stores the original WAV files, so recordings survive redeploys without putting large audio blobs in the database.

PostgreSQL stores users, scripts, sessions, reset tokens, recording metadata, audio details, and best-take choices. The volume stores only the original WAV files.

After deploying, generate one public Railway domain. The app and API are available on the same domain:

```text
https://your-service.up.railway.app
https://your-service.up.railway.app/api/health
```

No `NEXT_PUBLIC_API_BASE_URL` is needed for this single-container deployment because the frontend calls `/api` on the same origin.

Admin sign-in uses `ADMIN_EMAIL` and `ADMIN_PASSWORD` from Variables. On each deploy, the bootstrap admin user (`id=admin`) is synced to those values in PostgreSQL. If login fails, confirm the Variables match what you type (email is case-insensitive) and redeploy after changing them.

If Railway shows a 502 or "Application failed to respond", check these first:

- In **Settings → Networking → Public networking**, set the target port to **8080** (or remove a custom port so Railway uses the container `EXPOSE` port). Do **not** use **3000** there: port 3000 is only for the internal Next.js process behind nginx.
- The Railway service root directory should be empty / repo root, not `backend` or `frontend`.
- The Railway service should use the root `Dockerfile` and root `railway.toml`.
- Do not set `PORT` manually in Variables; Railway provides it automatically.
- Make sure `APP_ENV=production`, `DATABASE_URL`, `SECRET_KEY`, and `ADMIN_PASSWORD` are valid. The API intentionally refuses to start with weak production secrets or without PostgreSQL.
- `CORS_ORIGINS` should be the exact Railway app domain, for example `https://your-service.up.railway.app`.

### Optional Two-Service Deployment

The repository still includes `backend/Dockerfile`, `frontend/Dockerfile`, `backend/railway.toml`, and `frontend/railway.toml` if you prefer separate Railway services later.

For two services, deploy the backend with root directory:

```text
backend
```

Deploy the frontend with root directory:

```text
frontend
```

Then set this on the frontend:

```text
NEXT_PUBLIC_API_BASE_URL=https://your-api-service.up.railway.app
```

## Verification

```bash
cd backend && pytest
cd frontend && npm test && npm run lint && npm run build
```
