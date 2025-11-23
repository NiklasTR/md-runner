import subprocess
from pathlib import Path


def run_md_maybe_parallel(
    tmp_path: Path,
    project_root: Path,
    seq_name: str,
    pdb_dir: Path,
    num_parallel_procs: int = 1,
    env: dict = None,
):
    """
    Launch num_parallel_procs parallel MD processes under MPS.
    Each process gets its own output directory: base_output_dir/proc_{i}
    """

    jobs = []

    for i in range(num_parallel_procs):
        data_dir = tmp_path / f"parallel_md_{seq_name}" / f"proc_{i}"

        cmd = [
            "python",
            "src/generate_md.py",
            # Dynamic config options
            f"seq_name={seq_name}",
            f"paths.data_dir={data_dir}",
            f"pdb_dir={pdb_dir}",
            # Static MD parameters
            "warmup_steps=0",  # The way we benchmark, we don't need an equilibration phase
            "frame_interval=1000",  # 1ps per frame
            "time_ns=0.010",  # 100ps total simulation time
            "frames_per_chunk=100",  # Save 1000 frames at a time
        ]

        p = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        jobs.append(p)

    # Wait for all jobs
    for p in jobs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(
                f"MD job failed (exit {p.returncode}).\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}",
            )
