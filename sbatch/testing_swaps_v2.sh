#!/bin/bash
#SBATCH -J remd_tune
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --partition=long,main,unkillable
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 12:00:00
#SBATCH --array=0-10   # 6 sequences × 4 mods = 24 jobs

N_STATES_MOD=2

seq_idx=$(( SLURM_ARRAY_TASK_ID / N_STATES_MOD ))
n_states_mod=$(( SLURM_ARRAY_TASK_ID % N_STATES_MOD ))

echo "task=${SLURM_ARRAY_TASK_ID} seq_idx=${seq_idx} n_states_mod=${n_states_mod}"

conda activate md-runner

python src/pdb_to_sim/remd.py \
  seq_filename=sequences/test_sequences.txt \
  seq_idx=${seq_idx} \
  n_states_mod=${n_states_mod} \
  n_states="auto"
