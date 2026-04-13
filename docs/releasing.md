# Releasing

Releases are cut by pushing a tag matching `v*` (PEP 440 version preceded by
`v`, e.g. `v0.1.0`). The `release.yml` workflow then:

1. Builds an sdist + wheel with `python -m build` (hatchling backend).
2. Verifies the tag matches `project.version` in `pyproject.toml` — a
   mismatch fails the workflow before anything is published.
3. Publishes both distributions to PyPI via **trusted publishing (OIDC)** —
   no API tokens are stored in GitHub Secrets.
4. Attaches the same distributions to the GitHub Release and generates
   release notes from the commit history.

## One-time setup

PyPI's trusted publisher needs to be registered **before the first release**.
This is a manual step (~5 minutes) outside the workflow:

1. Create the `metalgrow` project on PyPI (or claim the name if it's
   available). If you haven't published a release yet, use PyPI's
   ["pending publisher" flow](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/):
   - PyPI → Your account → Publishing → Add a new pending publisher.
   - Owner: `joaodotwork` · Repository: `metalgrow` · Workflow:
     `release.yml` · Environment: `pypi`.
2. In the GitHub repo, create an environment named `pypi` (Settings →
   Environments → New environment). No secrets needed — OIDC handles auth.
   Consider adding a required reviewer if you want a manual gate before
   every publish.

After the first successful publish the pending publisher converts to a
regular trusted publisher.

## Cutting a release

Assuming `main` is green and you're ready to tag:

```bash
# 1. Bump the version
#    pyproject.toml:  version = "0.1.0"
$EDITOR pyproject.toml

# 2. Update CHANGELOG.md with the new section
$EDITOR CHANGELOG.md

# 3. Commit + tag on main
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v0.1.0"
git tag -a v0.1.0 -m "v0.1.0"
git push origin main v0.1.0
```

The workflow picks up the tag push and handles the rest. Watch it on the
Actions tab; if the version-check step fails, fix `pyproject.toml`, delete
and recreate the tag, and push again:

```bash
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
# fix, commit, re-tag, re-push
```

## Rollback

PyPI releases can't be deleted, only yanked. If a published version is
broken:

```bash
# On PyPI: Manage project → Releases → Yank (with reason).
# Then cut a fixed patch release (v0.1.1).
```

The GitHub Release can be deleted or edited freely from the web UI.
