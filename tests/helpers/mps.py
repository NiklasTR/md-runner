import os
import subprocess
import tempfile
import shutil
from contextlib import contextmanager


@contextmanager
def mps_context():
    """
    Start CUDA MPS, yield environment, then clean up.
    """

    # Create directories
    mps_dir = tempfile.mkdtemp(prefix="mps_")
    pipe_dir = os.path.join(mps_dir, "pipe")
    log_dir = os.path.join(mps_dir, "log")

    os.makedirs(pipe_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set the environment for subprocesses
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    env["CUDA_MPS_LOG_DIRECTORY"] = log_dir

    # Start the MPS control daemon
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)

    try:
        yield env  # This environment is used by all child processes
    finally:
        # Tell MPS server to quit
        subprocess.run(
            "echo quit | nvidia-cuda-mps-control",
            shell=True,
            env=env,
        )
        shutil.rmtree(mps_dir)
