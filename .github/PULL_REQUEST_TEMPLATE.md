## Summary

<!-- What changed and why. -->

## Benchmark integrity

Per [CONTRIBUTING.md](../CONTRIBUTING.md), the generator/judge model, `top_k`,
and `chunk_size` are frozen.

- [ ] This PR does not change a frozen benchmark knob, **or** the change is disclosed and justified above.
- [ ] If the change is results-affecting, I re-ran the relevant sample mode and noted the impact.

## Checks

- [ ] `prek run --all-files` passes (ruff check + format + ty)
- [ ] `uv run pytest -m "not integration and not slow" -n auto` passes
