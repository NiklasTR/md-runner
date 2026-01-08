#!/bin/bash
#SBATCH -J remd_waitn_dryrun
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=1G
#SBATCH -t 00:10:00
#SBATCH -c 1
#SBATCH --array=0-9
#SBATCH --open-mode=append

set -euo pipefail

mkdir -p watch_folder

SEQ_FILE="sequences/oligoREMD.txt"
TOTAL_PER_JOB=20
MAX_CONCURRENT=4

echo "Node: $HOSTNAME"
echo "ArrayTaskID: ${SLURM_ARRAY_TASK_ID:-no_array}"

# Build filtered sequence list
mapfile -t SEQS < <(grep -vE '^\s*$|^\s*#' "$SEQ_FILE")
NSEQ=${#SEQS[@]}

BASE_IDX=$(( SLURM_ARRAY_TASK_ID * TOTAL_PER_JOB ))

echo "TOTAL_PER_JOB=$TOTAL_PER_JOB"
echo "MAX_CONCURRENT=$MAX_CONCURRENT"
echo "NSEQ=$NSEQ"
echo "BASE_IDX=$BASE_IDX"
echo "----"

running=0

fake_work() {
  local idx=$1
  local task=$2
  echo "START  task=$task idx=$idx  pid=$$"
  sleep 10
  echo "END    task=$task idx=$idx  pid=$$"
}

for ((k=0; k<TOTAL_PER_JOB; k++)); do
  IDX=$(( BASE_IDX + k ))

  if (( IDX >= NSEQ )); then
    echo "IDX=$IDX out of range; stopping."
    break
  fi

  fake_work "$IDX" "$SLURM_ARRAY_TASK_ID" &

  running=$(( running + 1 ))
  echo "Launched idx=$IDX  (running=$running)"

  if (( running >= MAX_CONCURRENT )); then
    echo "(waiting…) running=$running"
    wait -n
    running=$(( running - 1 ))
    echo "(resumed) running=$running"
  fi
done

echo "Waiting for remaining $running job(s)…"
wait

echo "All fake work completed."
