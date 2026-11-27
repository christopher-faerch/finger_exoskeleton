"""run_emg.py --push: one session folder is committed and pushed, nothing else."""

import subprocess
from pathlib import Path

from source.publish import push_session


def git(folder: Path, *args: str) -> str:
    """Run git in folder and return its output; fails the test on a git error."""
    return subprocess.run(["git", *args], cwd=folder, capture_output=True, text=True,
                          check=True).stdout


def clone(remote: Path, folder: Path) -> Path:
    """A clone of remote with a committer set."""
    git(remote.parent, "clone", "-q", str(remote), str(folder))
    git(folder, "config", "user.name", "Test")
    git(folder, "config", "user.email", "test@example.com")
    return folder


def make_remote(tmp_path: Path) -> Path:
    """A bare remote with one commit on main."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    first = clone(remote, tmp_path / "first")
    (first / "README.md").write_text("x\n")
    git(first, "add", "README.md")
    git(first, "commit", "-q", "-m", "start")
    git(first, "push", "-q", "origin", "main")
    return remote


def session_in(checkout: Path, name: str) -> Path:
    """A session folder with an emg.csv in checkout/measurements."""
    folder = checkout / "measurements" / name
    folder.mkdir(parents=True)
    (folder / "emg.csv").write_text("time_s,flexor_mV\n0,0\n")
    return folder


def test_push_commits_only_the_session(tmp_path: Path) -> None:
    """The session reaches the remote; another staged change is not committed with it."""
    remote = make_remote(tmp_path)
    bench = clone(remote, tmp_path / "bench")
    (bench / "notes.txt").write_text("not for the commit\n")
    git(bench, "add", "notes.txt")
    folder = session_in(bench, "20261127-120000_trigno_pos-dsp")
    messages: list[str] = []
    assert push_session(folder, messages.append)
    files = git(remote, "ls-tree", "-r", "--name-only", "main").split()
    assert "measurements/20261127-120000_trigno_pos-dsp/emg.csv" in files
    assert "notes.txt" not in files
    assert "Recording pushed" in messages[-1]


def test_push_puts_the_session_on_top_of_new_remote_commits(tmp_path: Path) -> None:
    """When someone else pushed first, the session is rebased onto it and pushed."""
    remote = make_remote(tmp_path)
    bench = clone(remote, tmp_path / "bench")
    other = clone(remote, tmp_path / "other")
    (other / "code.py").write_text("print(1)\n")
    git(other, "add", "code.py")
    git(other, "commit", "-q", "-m", "code")
    git(other, "push", "-q")
    assert push_session(session_in(bench, "20261127-120001_trigno_vel-gru"), lambda text: None)
    files = git(remote, "ls-tree", "-r", "--name-only", "main").split()
    assert "code.py" in files
    assert "measurements/20261127-120001_trigno_vel-gru/emg.csv" in files


def test_outside_a_repository_nothing_happens(tmp_path: Path) -> None:
    """A session outside git is reported, not pushed."""
    folder = tmp_path / "session"
    folder.mkdir()
    messages: list[str] = []
    assert not push_session(folder, messages.append)
    assert "not in a git repository" in messages[0]
