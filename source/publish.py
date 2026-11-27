"""
Commit one session folder and push it (run_emg.py --push).

Only that folder is committed, whatever else is changed or staged in the checkout. If
the push is rejected because the remote has new commits, they are fetched and the
recording is put on top (git pull --rebase) before one more push. Anything that fails
is reported and left as it is: the recording is never lost, at worst it is only
committed locally and `git push` has to be run by hand.

The repository is PUBLIC: session labels must never contain names (use p1, p2, ...).
"""

import subprocess
from collections.abc import Callable
from pathlib import Path


def git(folder: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in folder; never raises for a git error, the caller reads returncode."""
    return subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True,
                          check=False)


def commit_session(folder: Path) -> tuple[Path | None, str]:
    """Commit only the session folder; (repository root, "") or (None, why it failed)."""
    top = git(folder, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return None, f"{folder} is not in a git repository"
    root = Path(top.stdout.strip())
    path = str(folder.relative_to(root.resolve()))
    added = git(root, "add", "--", path)
    if added.returncode != 0:
        return None, f"git add failed: {added.stderr.strip()}"
    committed = git(root, "commit", "-m", f"Recording {folder.name}", "--", path)
    if committed.returncode != 0:
        return None, f"git commit failed: {(committed.stdout + committed.stderr).strip()}"
    return root, ""


def push_session(folder: Path, message: Callable[[str], None] = print) -> bool:
    """Commit the session folder and push it; True if it reached the remote."""
    root, problem = commit_session(folder.resolve())
    if root is None:
        message(f"Not pushed: {problem}")
        return False
    if git(root, "push").returncode == 0:
        message(f"Recording pushed: {folder.name}")
        return True
    rebased = git(root, "pull", "--rebase", "--autostash")
    if rebased.returncode != 0:
        git(root, "rebase", "--abort")
        message(f"Recording committed but NOT pushed (pull failed: {rebased.stderr.strip()}). "
                "Run git pull and git push by hand.")
        return False
    pushed = git(root, "push")
    if pushed.returncode != 0:
        message(f"Recording committed but NOT pushed ({pushed.stderr.strip()}). "
                "Run git push by hand.")
        return False
    message(f"Recording pushed: {folder.name}")
    return True
