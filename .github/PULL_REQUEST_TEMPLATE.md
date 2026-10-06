## Summary

What this PR changes and why.

## Related issue

Closes #<!-- issue number -->

## Testing

The commands you ran, and anything a reviewer should run to reproduce.

- [ ] `pytest tests/` passes (and on free-threaded 3.14t, if concurrency changed)
- [ ] `ruff check`, `ruff format --check`, and `mypy membrane` pass
- [ ] `tools/check_naming.py` and `tools/check_docstrings.py` pass
- [ ] New or changed behavior is covered by tests; the coverage gate holds

## Checklist

- [ ] `CHANGELOG.md` is updated under `[Unreleased]` (with **Breaking** where flags or APIs change)
- [ ] `README.md` and `docs/` are updated if behavior or setup changed
