from pathlib import Path

import pytest

from tests.helpers.run_sh_command import run_sh_script


@pytest.fixture(scope="session")
def benchmark_pdb_dir(shared_tmp_path: Path) -> Path:
    """Generate PDB files for benchmark sequences.

    Args:
        shared_tmp_path: Session-scoped temporary directory path.

    Returns:
        Path to directory containing generated PDB files.
    """
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    from src.seq_to_pdb import seq_to_pdb
    from tests.helpers.utils import compose_config

    GlobalHydra.instance().clear()

    # Generate PDB files for all benchmark sequences
    seq_file = Path(__file__).parent / "test_benchmark_sequences.txt"
    cfg = compose_config(
        config_name="seq_to_pdb",
        overrides=[f"seq_filename={seq_file}"],
    )

    with open_dict(cfg):
        cfg.paths.data_dir = str(shared_tmp_path / "benchmark_data")
        cfg.paths.log_dir = str(shared_tmp_path / "benchmark_logs")
        cfg.paths.work_dir = str(Path.cwd())

    seq_to_pdb(cfg)

    pdb_dir = Path(cfg.paths.data_dir) / "pdbs"

    GlobalHydra.instance().clear()

    return pdb_dir


@pytest.mark.forked  # prevents OpenMM/tLEaP issues
@pytest.mark.parametrize("procs_per_gpu", [2, 4, 8])
@pytest.mark.benchmark
def test_benchmark_mps_parallel(
    benchmark,
    benchmark_pdb_dir: Path,
    tmp_path: Path,
    procs_per_gpu: int,
) -> None:
    """Benchmark MPS parallel execution with different process counts.

    Each run uses a function-level tmp_path for its output directory to avoid detecting
    previous runs as completed. PDB files are shared (session-scoped).

    Args:
        benchmark: pytest-benchmark fixture.
        benchmark_pdb_dir: Directory containing PDB files for benchmark sequences (shared).
        tmp_path: Function-level temporary directory for this test run's MD output.
        procs_per_gpu: Number of processes to run per GPU.
    """
    import os
    import sh

    # Set up paths - each benchmark run gets its own tmp_path for MD output
    seq_file = Path(__file__).parent / "test_benchmark_sequences.txt"
    script_path = Path(__file__).parent / "helpers" / "generate_md_mps_local.sh"
    project_root = Path(__file__).parent.parent

    # Use tmp_path for this benchmark run's MD output directory
    # This ensures each parallel configuration uses a separate output directory
    # so MD generation doesn't detect previous runs as completed
    unique_data_dir = tmp_path / "md_output"

    # Run the benchmark
    def run_benchmark():
        original_cwd = Path.cwd()

        try:
            # Change to project root so paths work correctly
            sh.cd(project_root)

            # Set up environment variables for generate_md
            os.environ["HYDRA_CONFIG_PATH"] = str(project_root / "configs")

            # Run the script with procs_per_gpu, seq_file, array task ID 0, pdb_dir, and unique data_dir
            # The unique data_dir (from tmp_path) ensures each benchmark run doesn't detect previous runs as completed
            run_sh_script(
                [
                    str(script_path),
                    str(procs_per_gpu),
                    str(seq_file),
                    "0",
                    str(benchmark_pdb_dir),
                    str(unique_data_dir),
                ],
            )
        finally:
            sh.cd(original_cwd)

    benchmark(run_benchmark)
