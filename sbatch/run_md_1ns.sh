#!/bin/bash
#SBATCH --job-name=md-1ns
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --output=run_md_1ns_%j.out
#SBATCH --error=run_md_1ns_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=16G

# Run 100 ns MD for each example sequence. Submit from pathfinder repo root:
#   sbatch packages/md-runner/sbatch/run_md_1ns.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
cd packages/md-runner

export PROJECT_ROOT="$(pwd)"
export SCRATCH_DIR="${SCRATCH_DIR:-./data}"

nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 30 > gpu_util_${SLURM_JOB_ID}.log 2>&1 &
NVSMI_PID=$!
trap "kill $NVSMI_PID 2>/dev/null" EXIT

for seq in AA ARIP GYDPETGTWG; do
  echo "Running 1 ns for $seq..."
  python src/pdb_to_sim/md.py seq_name="$seq" platform=gpu time_ns=100.0
done
echo "Done."
