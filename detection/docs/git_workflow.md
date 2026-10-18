# Team workflow and machine handoff

Use this guide to share a focused change with the datathon team or move the demo to another computer. Run Git commands from the repository root.

## Share a source change

Check the current branch and working tree before doing anything:

```bash
git branch --show-current
git status --short
git diff
```

If you are already on the intended team branch, stay there. Otherwise create a descriptive branch with `git switch -c your-branch-name`. Do not discard local edits to switch branches. Coordinate shared files with teammates before editing them.

After making a change, run the [root checks](../../README.md#check-a-team-change). Stage the intended files and review them before committing:

```bash
git add --patch
git diff --cached
git commit -m "Describe the resulting change"
git push -u origin HEAD
```

For new files, name them explicitly with `git add path/to/file`; `--patch` does not include untracked files by default. Share the branch/commit and test results with the team; use a pull request if that is the team's review process. Nothing in these instructions automatically merges or pushes changes.

## Start on another computer

1. Install Git and the platform's supported Python/native stack.
2. Clone the team's repository or update an existing checkout.
3. Select the agreed branch/commit and inspect `git status`.
4. Create a new environment using the destination [platform guide](../../README.md#start-here).
5. Generate/train locally or transfer the dataset/model files below.
6. Run dependency/native checks, then test the model/API before the presentation.

For a first checkout, replace the repository URL placeholder with the team's Git hosting URL, then enter the new folder:

```bash
git clone YOUR_TEAM_REPOSITORY_URL UNO-Datathon-Team
cd UNO-Datathon-Team
```

For an existing checkout with no uncommitted work:

```bash
git fetch origin
git switch your-branch-name
git pull --ff-only
git status --short
```

Replace the branch placeholder with the real team branch. If it exists only remotely, create the local tracking branch with `git switch --track origin/your-branch-name`. If pull cannot fast-forward, stop and reconcile histories with the team; do not use a destructive reset.

A branch change does not change `.venv`, ignored datasets, or saved models. Reinstall after dependency declarations change. Rebuild the environment after changing Python, platform, or checkout location; see [environment recovery](../../docs/environment.md).

## What to transfer

| Item | How to handle it |
| --- | --- |
| Source/configuration/docs | Share through the agreed Git branch/commit. |
| `.venv`, installed packages, executable launchers | Recreate on the destination; never copy. |
| Dataset | Copy the full simulation folder with manifest and all request/truth splits, or regenerate with the same configuration and runtime. |
| Trained detector | Copy the whole model directory. Keep `model.pt`, `tree.json`, `metadata.json`, and reports from one training run together. |
| ONNX graph | Include its manifest and matching source model/metadata. Recheck numerical parity. |
| Compiled `.so`/native libraries | Rebuild for the target architecture/runtime; they are not portable across Windows/x86/s390x. |
| Demo results | Transfer the complete run folder, including labels, requests, responses, timings, manifest, and summary. |

Use the team's approved file-transfer method for ignored artifacts; Git does not bring them along. Store data under a `data/` directory and models/reports under `artifacts/`. Keep a manifest of the source commit, model/config version, Python/native stack or image digest, and test results. A moved model may be portable, but dependency versions and numerical behavior still need checking.

When copying from a laptop to LinuxONE, compare predictions on a fixed ordered request stream using fresh state on both sides. Training on the laptop and serving on IBM is possible only after that compatibility check. Do not combine model files from different runs or declare hardware parity from a successful file copy.

## Use the IBM API from a laptop

The simplest arrangement is to run both the API and producer on IBM through two SSH terminals. For a local browser or local producer, forward the IBM host's loopback port. Replace the user/host placeholders:

```bash
ssh -L 8000:127.0.0.1:8000 your-user@your-ibm-host
```

Leave the SSH session open. Start the API on IBM's loopback port, or use the loopback-published container in the [LinuxONE guide](../../docs/platforms/linuxone.md). On the laptop, open `http://127.0.0.1:8000/health` and run the local demo with `--url http://127.0.0.1:8000`. The laptop needs only the producer's runtime to send traffic; the trained model stays on IBM. Keep host clocks synchronized for current-time demos.

If laptop port 8000 is occupied, forward a free port instead, for example `ssh -L 8001:127.0.0.1:8000 your-user@your-ibm-host`, then use port 8001 in the browser/client. Port forwarding requires the team's authorized SSH access; no public API binding is needed.

## Before presenting

- Prepare the dataset and complete model before the session.
- Run native checks, tests, and a short demo on the actual presentation machine.
- Start a fresh API, confirm `ready: true` and zero processed requests, then start one producer.
- Keep model/data paths and ports consistent across terminals.
- Restart the API before switching between independent demos/replays.
- Preserve the result folder and describe synthetic metrics and hardware measurements accurately.
