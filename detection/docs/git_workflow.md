# Sharing changes

Run Git commands from the repository root. Use a focused branch and review the
changes before committing:

```bash
git status --short
git switch -c cleanup/describe-your-change
git diff
```

Run the [development checks](../../CONTRIBUTING.md#code-quality), then stage the
files you intend to share. Review the staged diff before creating a commit:

```bash
git add --patch
git diff --cached
git commit -m "Describe the change"
git push -u origin HEAD
```

Open a pull request describing the problem, resulting behavior, and validation.
The repository's CI checks formatting, lint, and tests on Linux and Windows.

Generated datasets, model artifacts, reports, and virtual environments are
ignored. Keep generated output under `data/` or `artifacts/`, and follow the
[root guide](../../README.md) to reproduce a demo from a fresh checkout.
