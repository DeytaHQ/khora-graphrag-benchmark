# Contributing

Thanks for your interest in improving the Khora GraphRAG-Bench harness.

## Development setup

This project uses [`uv`](https://github.com/astral-sh/uv) for environment and
dependency management (matching the `khora` core repo).

```bash
uv sync --extra dev          # create .venv and install runtime + dev deps
uv run pre-commit install    # optional: run the hooks on every commit
```

The benchmark pins a specific `khora` release for reproducibility (see
`pyproject.toml`). To evaluate a different build, override at install time:

```bash
KHORA_SPEC="khora[accel] @ file:///path/to/local/khora" uv sync --extra dev
```

## Checks (run before pushing)

These are the same checks CI runs (`.github/workflows/ci.yml`):

```bash
uv run ruff check src/ tests/         # lint
uv run ruff format --check src/ tests/ # format
uv run ty check src/                   # type check
uv run pytest -m "not integration and not slow" -n auto  # unit tests + coverage
```

## Tests

- **Unit tests** (`tests/unit/`) must run with no external services - no
  Postgres, Neo4j, network, or API keys. Mock `khora`, `litellm`, and any
  storage. These run in CI on every push/PR.
- **Integration tests** (marker `integration`) may require Docker services
  (`docker compose up -d`) and live khora; they are excluded from the default
  run. Mark them `@pytest.mark.integration`.
- Coverage floor is enforced (`--cov-fail-under` in `pyproject.toml`); raise it
  as coverage improves, do not lower it.

## Benchmark integrity (please read)

This harness exists to produce **honest, reproducible** GraphRAG-Bench numbers.
When changing the adapter or harness:

- The benchmark-frozen knobs - generator/judge model (`gpt-4o-mini`), `top_k`,
  and `chunk_size` - must not be quietly changed to inflate scores. Method-side
  changes (retrieval architecture, reranking, graph algorithms, prompting) are
  fair game.
- Any deviation from the GraphRAG-Bench protocol must be disclosed in the
  results/README, not hidden.
- Report results from `--sample medium` or `full`; `--sample small` is
  noise-limited and not a reportable signal.

## Style

- Python 3.13, `ruff` for lint+format (line length 120, double quotes), `ty`
  for type checking.
- Prefer simple, readable code; only add error handling at system boundaries.
- Conventional-commit-style messages (`feat:`, `fix:`, `test:`, `docs:`,
  `chore:`) explaining the *why*.
