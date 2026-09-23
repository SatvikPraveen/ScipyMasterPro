# CI/CD

| Workflow | Trigger | Jobs |
| --- | --- | --- |
| `ci.yml` | push / PR | lint, advisory static analysis, tests on Linux/macOS/Windows and Python 3.11-3.13 (+3.14 experimental), notebook execution, package build, docs build, `All checks passed` gate |
| `docker.yml` | push to main, tags, Docker-related PRs | build, test Streamlit and JupyterLab services, run tests inside the image, Compose check, multi-arch push to GHCR, Trivy scan |
| `docs.yml` | push to main touching docs or utils | build MkDocs site and deploy to GitHub Pages |
| `codeql.yml` | push / PR / weekly | CodeQL security analysis |
| `release.yml` | `v*.*.*` tag | verify tag matches `pyproject.toml`, test, build, GitHub Release with artifacts, optional PyPI trusted publishing |

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `docs/CHANGELOG.md`.
2. Merge to `main`.
3. Tag and push:

    ```bash
    git tag v1.2.0 && git push origin v1.2.0
    ```

The release workflow refuses to run if the tag and the package version disagree.

To publish on PyPI, create a `pypi` environment in the repository settings, register the
trusted publisher on PyPI, and set the repository variable `PUBLISH_TO_PYPI=true`.

## Dependabot

Weekly grouped updates for pip, Docker and GitHub Actions land as pull requests and are
validated by the full CI pipeline.
