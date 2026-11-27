"""
train_gru.py, evaluate_gru.py and run_emg.py --method gru from start to end (needs PyTorch).

SOFTWARE TEST ONLY: everything here runs on made-up data. It shows that the
pipeline runs, not that the model recognizes real muscle activity or tremor.
"""

from pathlib import Path

import pytest

import run_emg

torch = pytest.importorskip("torch")

# pylint: disable=wrong-import-position
import evaluate_gru  # noqa: E402
import train_gru  # noqa: E402
from source.gru.model import load_checkpoint  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECORDING = ROOT / "measurements" / "20261105-144738_trigno_to-lange-greb"
REPLAY = ["--source", "recorded", "--file", str(RECORDING), "--headless", "--no-loop",
          "--calibration", str(ROOT / "calibration_replay.json"), "--method", "gru"]


@pytest.fixture(name="smoke_model", scope="module")
def fixture_smoke_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A model trained for 2 epochs on made-up data (one sensor)."""
    path = tmp_path_factory.mktemp("gru") / "smoke.pt"
    assert train_gru.main(["--smoke-test", "--epochs", "2", "--out", str(path)]) == 0
    return path


def test_smoke_training_writes_a_model_marked_synthetic(smoke_model: Path) -> None:
    """The model file exists, is for one flexor sensor and says it is synthetic."""
    config = load_checkpoint(smoke_model).config
    assert config.channel_roles == ["flexor"] and config.data_kind == "synthetic"
    assert len(config.training_groups) == 6 and "test" in config.metrics
    assert smoke_model.with_suffix(".json").is_file()


def test_evaluation_runs_and_leaves_out_training_recordings(
        smoke_model: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """evaluate_gru.py prints both methods and warns that the data is synthetic."""
    assert evaluate_gru.main(["--model", str(smoke_model), "--smoke-test"]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC DATA" in output and "GRU tremor" in output and "DSP tremor" in output


def test_run_emg_uses_the_gru_on_a_replay(smoke_model: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    """--method gru with a model: loaded, warmed up and in use; the run ends normally."""
    assert run_emg.main(REPLAY + ["--model", str(smoke_model), "--duration", "4"]) == 0
    output = capsys.readouterr().out
    assert "SYNTHETIC data: software test only" in output
    assert "GRU estimates in use" in output and "GRU grasp + DSP tremor: intention" in output


def test_a_synthetic_model_is_not_used_with_the_motor(
        smoke_model: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With --motor a model trained on made-up data is refused; the DSP drives the motor."""
    monkeypatch.setattr(run_emg, "connect_motor", lambda port, **_: (_ for _ in ()).throw(
        RuntimeError("stop before the motor")))
    with pytest.raises(RuntimeError, match="stop before the motor"):
        run_emg.main(REPLAY + ["--model", str(smoke_model), "--duration", "1", "--motor"])
    output = capsys.readouterr().out
    assert "not used with --motor" in output and "GRU estimates in use" not in output
