"""Entrypoint: exposes the FastAPI app (run with `uvicorn lofi.main:app`)."""

from lofi.api.app import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
