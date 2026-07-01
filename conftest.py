"""Root pytest configuration: load .env before any test runs."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (same directory as this file).
# override=False means existing shell env vars take precedence over .env values.
load_dotenv(Path(__file__).parent / ".env", override=False)
