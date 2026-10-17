# Keep this work off main

Use the existing `feat/ticket-scalper-detection` feature branch for the ticket-specific project. Review and commit work there before merging it into `main`.

Check your current branch and files:

```bash
git branch --show-current
git status --short
```

If you are already on `feat/ticket-scalper-detection`, continue below. Otherwise, after checking that your working files are ready to switch, select the existing branch:

```bash
git switch feat/ticket-scalper-detection
```

Uncommitted changes can follow a branch switch, and Git can refuse a switch that would overwrite them. Review and save changes on the appropriate branch instead of discarding them. A commit records the project state; creating a branch alone does not save uncommitted files.

Review and commit the project's source, tests, merchant-policy configuration, and documentation:

```bash
git diff -- README.md
git add .gitignore README.md pyproject.toml requirements.txt src tests examples docs
git diff --cached --stat
git commit -m "Build ticket-only scalper detection with cross-site event signals"
```

`git diff --cached` shows the staged file contents for a full review before committing. Confirm that only intended source, configuration, and documentation changes are staged. Ticket datasets, trained artifacts, and `.venv` stay local and are excluded by the project's ignore rules. A new checkout needs its own environment, team-supplied data, and training run; the repository does not supply transaction examples or a pretrained model. Follow the [README](../README.md) to train into `artifacts/ticket_model`.

Push this branch when you are ready to share it:

```bash
git push -u origin feat/ticket-scalper-detection
```

Pushing the named feature branch does not merge it into `main`. Open a pull request separately when the changes are ready for team review.

After committing, switch between branches with:

```bash
git switch main
git switch feat/ticket-scalper-detection
```

For future work starting from main, create a new branch before editing:

```bash
git switch main
git switch -c feat/your-next-change
```

Avoid force pushes or discarding working files to move the project between branches; neither is needed for this workflow.
