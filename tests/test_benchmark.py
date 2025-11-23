from pathlib import Path

import pytest

from tests.helpers.run_sh_command import run_sh_command


@pytest.mark.forked  # prevents OpenMM/tLEaP issues
def test_benchmark_generated_md_sequential(tmp_path: Path) -> None:
    """Test Optuna sweep with wandb logging and ddp sim.

    :param tmp_path: The temporary logging path.
    """
    command = [
        "logger=wandb",
    ]
    run_sh_command(command)
