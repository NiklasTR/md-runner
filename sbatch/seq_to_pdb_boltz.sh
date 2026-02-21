#!/bin/bash
#SBATCH --job-name=seq2pdb-boltz
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --output=seq_to_pdb_boltz_%j.out
#SBATCH --error=seq_to_pdb_boltz_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16G

# Submit from pathfinder repo root:
#   sbatch packages/md-runner/sbatch/seq_to_pdb_boltz.sh
#
# Runs Boltz-2 structure prediction then PDBFixer (hydrogens + protonation at pH 7).
# Requires: boltz, pdbfixer in .venv

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
cd packages/md-runner

export PROJECT_ROOT="$(pwd)"
export SCRATCH_DIR="${SCRATCH_DIR:-./data}"

python src/seq_to_pdb/boltz2.py seq_filename=sequences/example_sequences.txt
