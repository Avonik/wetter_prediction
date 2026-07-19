from typer.testing import CliRunner
from wetter.cli import app

runner = CliRunner()


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in [
        "pull-obs", "pull-forecasts", "build-dataset", "train", "evaluate", "report",
        "pull-runs", "build-hourly", "train-hourly", "forecast", "evaluate-rain",
    ]:
        assert cmd in result.output
