#!/bin/bash
python src/pdb_to_sim/md.py -m launcher=example \
seq_filename=sequences/example_sequences.txt \
seq_idx="range(0, 3)"
