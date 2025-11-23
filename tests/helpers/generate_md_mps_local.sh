#!/bin/bash

# Local version of generate_md_mps.sh for benchmarking
# Usage: ./generate_md_mps_local.sh <PROCS_PER_GPU> <SEQ_FILE> <SLURM_ARRAY_TASK_ID> [PDB_DIR] [DATA_DIR]

set -e

# Cleanup function to ensure MPS is stopped
cleanup() {
    if [ -n "$MPS_DIR" ] && [ -d "$MPS_DIR" ]; then
        echo quit | nvidia-cuda-mps-control 2>/dev/null || true
        echo "MPS stopped."
        rm -rf "$MPS_DIR" 2>/dev/null || true
        echo "Cleaned up MPS directory."
    fi
}

# Register cleanup function to run on exit
trap cleanup EXIT

PROCS_PER_GPU=${1:-4}
SEQ_FILE=${2:-"sequences/example_sequences.txt"}
SLURM_ARRAY_TASK_ID=${3:-0}
PDB_DIR=${4:-""}
DATA_DIR=${5:-""}

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"
echo "Processes per GPU: $PROCS_PER_GPU"
echo "Sequence file: $SEQ_FILE"

# ============================
# Count sequences
# ============================
NUM_LINES=$(wc -l < "$SEQ_FILE" | tr -d ' ')
echo "Total sequences: $NUM_LINES"

# ============================
# Start CUDA MPS
# ============================
# Create task-array-unique MPS directories
MPS_DIR=/tmp/$USER/mps_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"

export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

# Start the MPS server (ignore error if already running)
nvidia-cuda-mps-control -d 2>/dev/null || true
echo "MPS server started at $CUDA_MPS_PIPE_DIRECTORY"

# ============================
# Compute starting index
# ============================
BASE_IDX=$(( SLURM_ARRAY_TASK_ID * PROCS_PER_GPU ))

echo "Launching MD jobs from indices $BASE_IDX to $(( BASE_IDX + PROCS_PER_GPU - 1 ))"

# ============================
# Launch N processes
# ============================
for ((i=0; i<PROCS_PER_GPU; i++)); do
    IDX=$(( BASE_IDX + i ))
    if (( IDX >= NUM_LINES )); then
        echo "Index $IDX exceeds total sequences; skipping."
        continue
    fi

    echo "Launching process for seq_idx=$IDX"
    # Build command with optional parameters
    CMD="python src/generate_md.py seq_idx=$IDX seq_filename=$SEQ_FILE"
    if [ -n "$PDB_DIR" ]; then
        CMD="$CMD pdb_dir=$PDB_DIR"
    fi
    if [ -n "$DATA_DIR" ]; then
        CMD="$CMD paths.data_dir=$DATA_DIR"
    fi
    eval "$CMD &"
done

wait
echo "All MD jobs completed."

# Cleanup will be handled by trap on exit
echo "Done."

