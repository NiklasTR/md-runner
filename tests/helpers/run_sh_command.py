import subprocess
from typing import List, Union

import pytest


def run_sh_script(command: Union[List[str], str], cwd: Union[str, None] = None) -> None:
    """Execute shell scripts using subprocess.

    For bash scripts, can pass as a string (will be split) or as a list of arguments.

    :param command: A string path to a bash script (will be split by spaces),
                    or a list of arguments where the first element is the script path.
                    Use list format if paths contain spaces.
    :param cwd: Optional working directory for the subprocess.
    """
    try:
        if isinstance(command, str):
            # Bash script - split into command and args, then run with bash
            import shlex

            parts = shlex.split(command)
            result = subprocess.run(
                ["bash"] + parts,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            # Bash script passed as list of arguments
            result = subprocess.run(
                ["bash"] + command,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        msg = e.stderr if e.stderr else str(e)
        pytest.fail(msg=msg)
