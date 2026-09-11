# Release process

This page documents the procedure for cutting a new Membrane
release. Releases are tied to the version in ``pyproject.toml``
and to the ``vX.Y.Z`` git tag; the CI workflow
(``.github/workflows/release.yml``) publishes the wheel,
sdist, and ``ghcr.io/sachncs/membrane`` image on every tag.

## Versioning

Membrane follows [Semantic Versioning](https://semver.org/):

* **Major** — breaking wire-format or public-API changes.
  Requires a schema_version bump in
  :data:`membrane.serialization.SCHEMA_VERSION` and a
  migration window.
* **Minor** — backward-compatible feature additions.
* **Patch** — backward-compatible bug fixes.

## Pre-release checklist

- [ ] All issues targeted for this release are closed or
      explicitly deferred.
- [ ] The full test suite (`pytest tests/ -v`) is green on
      every supported Python version.
- [ ] The lint (`ruff check membrane/ tests/`) and type
      check (`mypy membrane/`) are green.
- [ ] The bench smoke (`pytest tests/bench/ -v`) is green.
- [ ] The chaos suite (`pytest tests/membrane/chaos -m chaos`)
      is green (or documented as self-hosted-runner only).
- [ ] `CHANGELOG.md` has a new section above the unreleased
      marker with the version, the date, and a "Breaking" /
      "Added" / "Fixed" / "Changed" block.

## Cut a release

```bash
# 1. Bump the version. Edit pyproject.toml:
#       version = "X.Y.Z"
#    and (for breaking changes) bump
#    membrane.serialization.SCHEMA_VERSION.

# 2. Update CHANGELOG.md: move the unreleased section to a
#    "## [X.Y.Z] - YYYY-MM-DD" heading.

# 3. Commit.
git add pyproject.toml CHANGELOG.md
git commit -m "release: vX.Y.Z"

# 4. Tag. The tag format is `vX.Y.Z`.
git tag -s vX.Y.Z -m "vX.Y.Z"

# 5. Push the tag. The release workflow runs from here.
git push origin master
git push origin vX.Y.Z
```

The CI workflow ``.github/workflows/release.yml`` picks up
the tag and:

1. Builds the sdist and wheel via ``python -m build``.
2. Publishes them to PyPI under ``membrane``.
3. Builds the Docker image and pushes it to
   ``ghcr.io/sachncs/membrane:vX.Y.Z`` and
   ``ghcr.io/sachncs/membrane:latest``.

## Post-release

- [ ] Verify the PyPI release is visible at
      https://pypi.org/project/membrane/.
- [ ] Verify the GHCR image is visible at
      https://github.com/sachncs/membrane/pkgs/container/membrane.
- [ ] Update the k8s StatefulSet image tag in
      ``deployment/k8s/statefulset.yaml`` to match the
      newly-published version.
- [ ] Smoke-test ``pip install membrane`` in a fresh venv
      and run ``python scripts/demo.py``.

## Rollback

If the release ships a regression:

1. Yank the PyPI release (``twine yank membrane==X.Y.Z``).
2. Delete the GHCR tag from
   ``ghcr.io/sachncs/membrane:vX.Y.Z``.
3. Cut a patch release (``X.Y.(Z+1)``) with the fix.

The wire format is forward-compatible within a major series
but **not** backward-compatible across majors; rolling back
to a prior major requires the operator to run
``tools/upgrade_v2_to_v5.py`` (or whichever one-shot
migration matches the destination version) before booting
the older image.
