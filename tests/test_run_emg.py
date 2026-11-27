"""run_emg.py from start to end: replay, recording, the fake TCU and the motor."""

import json
from pathlib import Path
from typing import Any

import pytest

import run_emg
import settings
from source.motor import connect_motor
from tests.fakes import FakeMotors

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"


def test_replay_with_recording(tmp_path: Path) -> None:
    """A headless replay with --out writes a session folder that can be replayed again."""
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.5", "--out", str(tmp_path), "--label", "test"])
    assert code == 0
    (folder,) = tmp_path.iterdir()
    metadata = json.loads((folder / "session.json").read_text())
    assert metadata["status"] == "complete" and metadata["samples_per_channel"] > 2000
    assert metadata["method"] == "dsp" and metadata["calibration"] is None  # too short to calibrate
    assert (folder / "emg.csv").read_text().startswith("time_s,FDS_mV,EDC_mV")


@pytest.mark.parametrize("control", ["position", "velocity"])
def test_motor_ends_open(monkeypatch: pytest.MonkeyPatch, control: str) -> None:
    """With --motor the motor runs during the replay and is back at open at the end."""
    made: list[FakeMotors] = []

    def fake_motors(**kwargs: Any) -> FakeMotors:
        made.append(FakeMotors(**kwargs))
        return made[0]

    def connect(port: str, **kwargs: Any) -> Any:
        assert kwargs["control"] == control
        return connect_motor(port, motors_class=fake_motors, **kwargs)

    monkeypatch.setattr(run_emg, "connect_motor", connect)
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.0", "--motor", "--control", control])
    assert code == 0
    assert made[0].sent and made[0].sent[-1] == settings.OPEN_POS


def test_fake_tcu() -> None:
    """--mock-tcu streams a recording through the real Trigno client."""
    code = run_emg.main(["--source", "trigno", "--mock-tcu", "--headless", "--duration", "1.5"])
    assert code == 0


def test_gru_without_a_model_file_runs_on_the_signal_processing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--method gru with a missing model file says so and runs on the signal processing."""
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "0.5", "--method", "gru",
                         "--model", str(tmp_path / "none.pt")])
    assert code == 0
    assert "GRU not available" in capsys.readouterr().out


def test_cues_are_shown_and_saved_as_labels(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    """--cues prints each cue when it starts and writes labels.csv into the session folder."""
    protocol = tmp_path / "protocol.csv"
    protocol.write_text("start_s,end_s,cue,intention,tremor\n"
                        "0,0.6,HVIL,0,0\n0.6,5,LUK HELT,1,\n", encoding="utf-8")
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.5", "--out", str(tmp_path / "out"),
                         "--cues", str(protocol)])
    assert code == 0
    output = capsys.readouterr().out
    assert output.index(">>> HVIL") < output.index(">>> LUK HELT")
    (folder,) = (tmp_path / "out").iterdir()
    labels = (folder / "labels.csv").read_text().splitlines()
    assert labels == ["start_s,end_s,intention,tremor,cue", "0,0.6,0,0,HVIL",
                      "0.6,5,1,,LUK HELT"]


def test_cues_need_a_recording() -> None:
    """--cues without --out is refused: the labels would have nowhere to go."""
    with pytest.raises(SystemExit):
        run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--cues", "x.csv"])


def test_start_delay_is_saved_and_used_again_for_a_replay(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--start-delay shifts cues and labels, is stored in session.json and reused on replay."""
    protocol = tmp_path / "protocol.csv"
    protocol.write_text("start_s,end_s,cue,intention,tremor\n0,5,HVIL,0,0\n", encoding="utf-8")
    code = run_emg.main(["--source", "recorded", "--file", str(RECORDING), "--headless",
                         "--duration", "1.5", "--out", str(tmp_path / "out"),
                         "--cues", str(protocol), "--start-delay", "0.7"])
    assert code == 0
    output = capsys.readouterr().out
    assert output.index(">>> VENT: starter om 1 s") < output.index(">>> HVIL")
    (folder,) = (tmp_path / "out").iterdir()
    labels = (folder / "labels.csv").read_text().splitlines()
    assert labels[1:] == ["0,0.7,,,VENT", "0.7,5.7,0,0,HVIL"]
    assert json.loads((folder / "session.json").read_text())["start_delay_s"] == 0.7

    args = run_emg.parse_args(["--source", "recorded", "--file", str(folder)])
    replay = run_emg.RecordedSource(folder)
    assert run_emg.start_delay(args, replay) == 0.7
    assert run_emg.start_delay(args, run_emg.RecordedSource(RECORDING)) == 0.0  # old recording
