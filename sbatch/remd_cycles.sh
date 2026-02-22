#!/bin/bash
#SBATCH --job-name=remd-cycles
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=remd_cycles_%A_%a.out
#SBATCH --error=remd_cycles_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00
#SBATCH --mem=16G
#SBATCH --array=0-15

# Submit from pathfinder repo root:
#   sbatch packages/md-runner/sbatch/remd_cycles.sh
#
# Runs 1us REMD for each cyclic peptide in example_cycles.txt.
# Each array task simulates one peptide (0-indexed by line number).
# Expects NPZ files in data/pdbs/ from a prior seq_to_pdb_boltz run.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
cd packages/md-runner

export PROJECT_ROOT="$(pwd)"
export SCRATCH_DIR="${SCRATCH_DIR:-./data}"

python -m src.pdb_to_sim.remd \
    seq_filename=sequences/example_cycles.txt \
    seq_idx=${SLURM_ARRAY_TASK_ID} \
    time_ns=1000.0 \
    n_states=auto
