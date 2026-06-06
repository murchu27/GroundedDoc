# FastAPI Migration Guide (Deprecated)

## APIRouter

Legacy applications should replace deprecated `Route` helpers with `APIRouter`.
The old `route()` decorator pattern is removed in current releases.

## Dependency Injection

The deprecated `dependency_overrides_provider` argument must be replaced with app-level overrides.

## Response Model

The legacy `response_model_exclude_unset` default behavior changed; migration requires explicit configuration.
