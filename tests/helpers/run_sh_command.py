from typing import List, Union

import pytest
import sh


def run_sh_script(command: Union[List[str], str]) -> None:
    """Execute shell scripts with `pytest` and `sh` package.

    For bash scripts, can pass as a string (will be split) or as a list of arguments.

    :param command: A string path to a bash script (will be split by spaces),
                    or a list of arguments where the first element is the script path.
                    Use list format if paths contain spaces.
    """
    msg = None
    try:
        if isinstance(command, str):
            # Bash script - split into command and args, then run with bash
            import shlex

            parts = shlex.split(command)
            sh.bash(parts)
        else:
            # Bash script passed as list of arguments
            sh.bash(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)
