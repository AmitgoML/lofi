# Lucy backend documentation

Backend-specific docs complement the workspace-level [`../../docs/`](../../docs/) tree.

| Doc | Purpose |
|-----|---------|
| [Module map](../../docs/architecture/README.md) | System architecture and `lucy/` module boundaries |
| [API contracts](../../docs/api/README.md) | Workflow + streaming endpoint spec |
| [Runbooks](../../docs/runbooks/README.md) | Pipeline and connector operations |
| [Open questions](../../docs/open-questions.md) | Week 0 blockers |

## Local development

See root [README.md](../README.md).

## Deployment

See [deploy.md](../docs/deploy.md) if present, or `terraform/` + `deploy.sh`.

## Tests

```sh
make test
```

New module import safety: `tests/test_scaffold_imports.py`
