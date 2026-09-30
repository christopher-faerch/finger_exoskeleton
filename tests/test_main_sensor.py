import pytest

import main_sensor


def test_main_prints_hello_world(capsys: pytest.CaptureFixture[str]) -> None:
    """main() prints the greeting to stdout."""
    main_sensor.main()
    assert capsys.readouterr().out == "Hello world\n"
