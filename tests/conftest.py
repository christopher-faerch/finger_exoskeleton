from dataclasses import replace

import matplotlib
import pytest

from source.activation import Calibration
from source.actuator import DryRunActuator
from source.config import AppConfig, load_config
from source.runtime import ControlCore, run_synchronously
from source.sensor import SyntheticSource

# Render figures off-screen so tests never open plot windows
matplotlib.use("Agg")


@pytest.fixture(name="config")
def fixture_config() -> AppConfig:
    """The project configuration, resolved for the synthetic source."""
    return load_config(source="synthetic")


def offline_synthetic(config: AppConfig) -> SyntheticSource:
    """A synthetic source that generates as fast as possible."""
    return SyntheticSource(replace(config.synthetic, realtime=False), config.channels)


@pytest.fixture(name="calibration", scope="session")
def fixture_calibration() -> Calibration:
    """A calibration recorded from the synthetic scenario (shared by the whole session)."""
    config = load_config(source="synthetic")
    source = offline_synthetic(config)
    source.start()
    core = ControlCore(config, source.sample_rate, DryRunActuator(float("inf")))
    run_synchronously(core, source, 6.0, arm_when_calibrated=False)
    assert core.calibrator.result is not None, core.calibrator.error
    return core.calibrator.result
