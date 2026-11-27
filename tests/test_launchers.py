"""The six launchers run_{position,velocity,threshold}_{dsp,rnn}.bat/.sh start run_emg.py right."""

import shlex
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import run_emg
from source.injection import INJECTIONS_FILE, read_episodes
from source.injection_evaluation import read_motor_log
from source.motor import connect_motor
from tests.fakes import FakeMotors

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"
LAUNCHERS = {f"run_{control}_{name}": (control, method)
             for control in ("position", "velocity", "threshold")
             for name, method in (("dsp", "dsp"), ("rnn", "gru"))}


def launcher_args(path: Path) -> list[str]:
    """The options a launcher gives run_emg.py (Windows paths made portable)."""
    text = path.read_text().replace("\\\n", " ")
    (line,) = [line for line in text.splitlines() if "run_emg.py" in line and "#" not in line
               and not line.lower().startswith("rem")]
    words = shlex.split(line.replace("\\", "/"))
    return [word for word in words[words.index("run_emg.py") + 1:]
            if word not in ("%*", "$@")]


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
@pytest.mark.parametrize("suffix", [".bat", ".sh"])
def test_launcher_options(name: str, suffix: str) -> None:
    """Each launcher drives the motor with its control and method, live and recorded."""
    args = run_emg.parse_args(launcher_args(ROOT / f"{name}{suffix}"))
    assert (args.control, args.method) == LAUNCHERS[name]
    assert args.motor and args.source == "trigno" and args.out == Path("measurements")
    assert not args.headless  # the window, where k switches the simulated tremor


@pytest.mark.parametrize("name", sorted(LAUNCHERS))
def test_launcher_runs_with_tremor(name: str, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """A launcher's options on a replay with a fake motor: runs, records the tremor, ends open."""
    made: list[FakeMotors] = []

    def fake_motors(**kwargs: Any) -> FakeMotors:
        made.append(FakeMotors(**kwargs))
        return made[0]

    monkeypatch.setattr(run_emg, "connect_motor", lambda port, **kwargs: connect_motor(
        port, motors_class=fake_motors, **kwargs))
    extra = ["--source", "recorded", "--file", str(RECORDING), "--headless", "--duration", "1.5",
             "--out", str(tmp_path), "--calibration", str(tmp_path / "calibration.json"),
             "--model", str(tmp_path / "missing.pt"), "--inject", "0.3:0.8"]
    assert run_emg.main(launcher_args(ROOT / f"{name}.sh") + extra) == 0
    (folder,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(read_episodes(folder / INJECTIONS_FILE)) == 1
    assert made[0].sent
    motor = read_motor_log(folder)  # the motor log, on the stream clock of emg.csv
    assert motor is not None and len(motor.times) > 10
    assert 0.0 <= motor.times[0] < motor.times[-1] <= 1.6
    assert np.all((motor.closure > -0.05) & (motor.closure < 1.05))
