.PHONY: start test test-watch clean-history deploy-stage

LOG_LEVEL ?= info
ENV ?= local
PORT ?= 8000
DB_FILE ?=

start:
	-kill -9 $$(lsof -tiTCP:$(PORT)) 2>/dev/null || true; APP_ENV=$(ENV) uv run uvicorn lucy.app:app --reload --log-level $(LOG_LEVEL) --port $(PORT)


test:
	PYTHONPATH=. APP_ENV=local uv run --all-groups pytest -q

test-watch:
	PYTHONPATH=. uv run --all-groups ptw -- -q


clean-history:
	@echo "No local SQLite history to clean."

deploy-stage:
	git push --force origin "$$(git branch --show-current)":stage

