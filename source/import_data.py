"""
Loads the sEMG hand gesture dataset (Ozdemir et al., Data in Brief 41 (2022) 107921,
docs/dataset_emg_hand_gestures.pdf).

The dataset contains 4 sEMG channels in mV, sampled at 2000 Hz and recorded
simultaneously from four forearm muscles with bipolar surface electrodes.
The channel mapping is given in Fig. 2 of the paper:

- CH1: Extensor carpi ulnaris (ECU)
- CH2: Flexor carpi ulnaris (FCU)
- CH3: Extensor carpi radialis (ECR)
- CH4: Flexor carpi radialis (FCR)

Each recording is 640 s: 5 cycles of 104 s with 30 s rest in between (Fig. 4).

Note: In the *_raw.csv files, CH4 (FCR) is not raw. It has no DC offset, a
50 Hz notch and no content below 5 Hz or above 500 Hz, i.e. it already went
through the paper's online filter (5-500 Hz band-pass + 50 Hz notch).
"""
import csv
from enum import Enum
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sEMG_online"


class Muscle(Enum):
    """Forearm muscle, valued by its column index in the CSV files (paper Fig. 2)."""
    ECU = 0
    FCU = 1
    ECR = 2
    FCR = 3


class EMGData:
    """A named EMG signal: a list of samples in the given unit."""

    class NoDataError(Exception):
        """Exception raised when no EMG data is avaliable."""

        def __init__(self, message: str):
            self.message = message
            super().__init__(self.message)

    def __init__(self, name: str, unit: str = "mV"):
        self.data: list[float] | None = None
        self.name = name
        self.unit = unit

    def set_name(self, name: str) -> None:
        """Set the name used to label this signal in plots."""
        self.name = name

    def get_data(self) -> list[float]:
        """Return the samples, or raise NoDataError if none are loaded."""
        if self.data is None:
            raise EMGData.NoDataError(f"No data in EMGData '{self.name}'")
        return self.data

    def append_element(self, ele: float) -> None:
        """Append one sample, creating the data list if it is empty."""
        if self.data is None:
            self.data = []
        self.data.append(ele)

    def clear(self) -> None:
        """Remove all samples."""
        if self.data is not None:
            self.data.clear()

    def load_from_file(self, name: str, muscle: Muscle) -> bool:
        """Load one muscle's channel from a CSV in DATA_PATH. Returns False if missing."""
        file_path: Path = DATA_PATH / name
        channel: int = muscle.value

        if not file_path.exists():
            return False

        data: list[float] = []

        with open(file_path, "r", newline="", encoding="utf-8") as file:
            for row in csv.reader(file):
                try:
                    data.append(float(row[channel]))
                except ValueError:
                    # Skip a header row, if the CSV has one
                    if data:
                        raise

        self.data = data
        return True


def resolve_data_file(name: str | Path) -> Path:
    """An absolute path is used as is; a bare name is looked up in DATA_PATH."""
    path = Path(name)
    return path if path.is_absolute() else DATA_PATH / path


def load_columns(name: str | Path, columns: list[int]) -> list[list[float]]:
    """
    Load several CSV columns in one pass, in the order given (one list per column).
    A non-numeric first row is skipped as a header, like EMGData.load_from_file.
    Raises FileNotFoundError if the file is missing.
    """
    file_path = resolve_data_file(name)
    data: list[list[float]] = [[] for _ in columns]

    with open(file_path, "r", newline="", encoding="utf-8") as file:
        for row_number, row in enumerate(csv.reader(file)):
            try:
                values = [float(row[column]) for column in columns]
            except ValueError:
                if row_number == 0:
                    continue
                raise
            for channel, value in zip(data, values):
                channel.append(value)

    return data
