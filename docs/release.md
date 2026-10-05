# Release process

This page documents the procedure for cutting a new Membrane
release. Releases are tied to the version in ``pyproject.toml``
and to the ``vX.Y.Z`` git tag. On every tag,
``.github/workflows/release.yml`` runs the full CI suite, then
attaches the wheel and sdist to a GitHub Release and pushes the
``ghcr.io/sachncs/membrane`` image (``linux/amd64`` and
``linux/arm64``).

Membrane is **not** published to PyPI: the ``membrane`` name there
belongs to an unrelated project.

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
- [ ] CI is green on ``master`` (the ``CI passed`` check covers
      lint (ruff, naming, docstrings), types, tests on Python 3.14, Redis integration,
      stress / chaos / bench smoke, security scans, the container
      smoke test, the site build, and docs links).
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

The release workflow then:

1. Runs the full CI workflow and checks that the tag matches the
   ``pyproject.toml`` version and that ``CHANGELOG.md`` has a
   matching section.
2. Builds the sdist and wheel and attaches them, with the
   CHANGELOG section as release notes, to a GitHub Release.
3. Builds the multi-arch image (with SBOM and provenance) and pushes
   ``ghcr.io/sachncs/membrane:X.Y.Z``, ``:X.Y``, and ``:latest``.

## Post-release

- [ ] Verify the GitHub Release and its assets at
      https://github.com/sachncs/membrane/releases.
- [ ] Verify the GHCR image is visible at
      https://github.com/sachncs/membrane/pkgs/container/membrane.
- [ ] Update the k8s StatefulSet image tag in
      ``deployment/k8s/statefulset.yaml`` to match the
      newly-published version.
- [ ] Smoke-test the release in a fresh venv:
      ``pip install "membrane[server] @ git+https://github.com/sachncs/membrane.git@vX.Y.Z"``,
      then follow the [Quickstart](getting-started.md).

## Rollback

If the release ships a regression:

1. Mark the GitHub Release as a pre-release (or delete it) so it
   is no longer the latest.
2. Re-point ``ghcr.io/sachncs/membrane:latest`` at the previous
   version, or delete the bad tag.
3. Cut a patch release (``X.Y.(Z+1)``) with the fix.

The wire format is forward-compatible within a major series
but **not** backward-compatible across majors; rolling back
to a prior major requires the operator to run
``tools/upgrade_v2_to_v5.py`` (or whichever one-shot
migration matches the destination version) before booting
the older image.
