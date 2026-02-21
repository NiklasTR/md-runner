# MD Runner

**MD Runner** is the molecular dynamics toolkit used to generate the [**ManyPeptidesMD**](https://huggingface.co/datasets/transferable-samplers/many-peptides-md) dataset introduced in our work:
[**Amortized Sampling with Transferable Normalizing Flows**](https://arxiv.org/abs/2508.18175) (NeurIPS 2025).

The codebase builds on and extends the MD simulation tools provided in [**TimeWarp**](https://github.com/microsoft/timewarp).

## Installation

### Option 1: Conda / Micromamba (recommended)

Requires [AmberTools](https://ambermd.org/AmberTools.php) for tLEaP-based sequence-to-PDB conversion.

```bash
micromamba env create -f environment.yaml
micromamba activate md-runner
```

### Option 2: uv

```bash
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[boltz]"
```

**Note:** AmberTools (for tLEaP) is not available via pip. Install it separately with conda (`conda install -c conda-forge ambertools`) if you need `src/seq_to_pdb/tleap.py`.

### Optional: Boltz-2 for structure prediction

To use the `src/seq_to_pdb/boltz2.py` module (Boltz-2–based structure prediction), install the optional dependency:

```bash
pip install boltz[cuda] -U
# or with uv:
uv pip install boltz[cuda]
```

## Workflow

### 1. Generate Initial PDB Structures from Sequences

We use **tLEaP** (AmberTools) to construct peptide initial conformations from amino-acid sequences.

To convert a set of sequences into PDB files:

```bash
python src/seq_to_pdb/tleap.py seq_filename=sequences/example_sequences.txt
```

### 2. Run Molecular Dynamics Simulations (Local)

You can perform a local test/benchmark MD simulation using:

```bash
python src/pdb_to_sim/md.py seq_name=AA
```

### 3. Run Molecular Dynamics Simulations (SLURM)

For large-scale dataset generation, use the [hydra-submitit-launcher](https://hydra.cc/docs/plugins/submitit_launcher/) plug-in. An example script is provide in:

```bash
./submitit/run-md.py
```

Each individual sequence in `seq_filename` will be launched as a separate SLURM job.

## Citation

If you use this codebase or [**ManyPeptidesMD**](https://huggingface.co/datasets/transferable-samplers/many-peptides-md), please cite our paper:

```
@misc{tan2025amortized
      title={Amortized Sampling with Transferable Normalizing Flows}, 
      author={Charlie B. Tan and Majdi Hassan and Leon Klein and Saifuddin Syed and Dominique Beaini and Michael M. Bronstein and Alexander Tong and Kirill Neklyudov},
      year={2025},
      eprint={2508.18175},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.18175}, 
}
```