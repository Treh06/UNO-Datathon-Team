# Keep the integrated project off main

The combined project uses `feature/data-generator`. It brings the original `feat/ticket-scalper-detection` work into the generator branch, with the detector in `detection/` and the generator in `generation_and_testing/`. The generator's original README is preserved.

Run Git commands from the repository root. Check the branch and files before committing:

```bash
git branch --show-current
git status --short
```

If needed, select the existing integration branch:

```bash
git switch feature/data-generator
```

Uncommitted files can follow a branch switch. Git can refuse a switch that would overwrite them. Review and commit your work on the appropriate branch before switching; creating a branch alone does not save files.

Review and save source, tests, configuration, and documentation:

```bash
git diff
git add .gitignore README.md pytest.ini detection generation_and_testing
git diff --cached --stat
git diff --cached
git commit -m "Update integrated ticket-scalper workflow"
git push -u origin feature/data-generator
```

Generated datasets, trained artifacts, and `.venv/` are ignored. Keep additional generated output in ignored directories too. A fresh checkout needs dependency installation and a training run; no transaction dataset or pretrained model is bundled. Follow the [repository README](../../README.md) for the combined workflow.

Pushing `feature/data-generator` does not merge it into `main`. Open a pull request separately when the team is ready. Neither force pushing nor discarding files is needed for this workflow.
