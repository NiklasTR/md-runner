import pytest
from pathlib import Path

from tests.helpers.mps import mps_context
from tests.helpers.run_md_maybe_parallel import run_md_maybe_parallel
import pytest

from tests.conftest import TEST_SEQUENCE

TOTAL_RUNS = 8
assert TOTAL_RUNS >= 1
assert TOTAL_RUNS > 0 and (TOTAL_RUNS & (TOTAL_RUNS - 1)) == 0, \
    "TOTAL_RUNS must be a power of 2"
PARALLEL_PROC_VALUES = [2 ** i for i in range(0, int(TOTAL_RUNS).bit_length())]  # [1, 2, ..., TOTAL_RUNS]

@pytest.mark.forked
# Only benchmark 1 (sequential) and 2 (naive parallel) processes without MPS.
# 2 is enough to see the large difference with MPS, hence we skip larger values here.
@pytest.mark.parametrize("num_parallel_procs", [1, 2])
@pytest.mark.benchmark
def test_benchmark_md_no_mps(
    benchmark,
    dir_with_pdb: Path,
    tmp_path: Path,
    num_parallel_procs: int,
):
    project_root = Path(__file__).parent.parent

    def run():
        for batch_id in range(TOTAL_RUNS):
            # By running without MPS env, we benchmark normal CUDA behavior.
            run_md_maybe_parallel(
                tmp_path=tmp_path / f"batch_{batch_id}",
                project_root=project_root,
                seq_name=TEST_SEQUENCE,
                pdb_dir=dir_with_pdb,
                num_parallel_procs=num_parallel_procs,
            )

    benchmark(run)


@pytest.mark.forked
@pytest.mark.parametrize("num_parallel_procs", PARALLEL_PROC_VALUES)
@pytest.mark.benchmark
def test_benchmark_md_mps(
    benchmark,
    dir_with_pdb: Path,
    tmp_path: Path,
    num_parallel_procs: int,
):
    project_root = Path(__file__).parent.parent
    total_runs = TOTAL_RUNS
    num_batches = total_runs // num_parallel_procs

    def run():
        for batch_id in range(num_batches):
            with mps_context() as env:
                run_md_maybe_parallel(
                    tmp_path=tmp_path / f"batch_{batch_id}",
                    env=env,
                    project_root=project_root,
                    seq_name=TEST_SEQUENCE,
                    pdb_dir=dir_with_pdb,
                    num_parallel_procs=num_parallel_procs,
                )

    benchmark(run)
