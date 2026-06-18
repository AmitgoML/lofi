# Lucy — Lofi AI Backend

FastAPI backend for Lucy agents, extended for the Lofi × GoML engagement:
orchestration, data pipeline, competitive intelligence, and evaluation.

See the workspace [README](../README.md) and [docs/architecture](../docs/architecture/README.md) for the full module map.

## Running Lucy App Locally

Follow the steps below to set up and run the Lucy application.

## 1. Create Your Environment File

Copy the example environment file:

```sh
cp .env-example .env
```

Update any required values inside `.env`.

## 2. Install Dependencies

### Python

Ensure Python is installed on your machine.

### UV Package Manager

Install **uv**:

```sh
pip install uv
```

### Sync Project Dependencies

Use uv to install all packages listed in the project:

```sh
uv sync
```

## 3. Start the Server

Start the application:

```sh
make start
```

The server will run at **[http://localhost:8000](http://localhost:8000)**.

