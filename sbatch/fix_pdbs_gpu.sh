#!/bin/bash
#SBATCH --job-name=fix-pdbs
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --output=fix_pdbs_%j.out
#SBATCH --error=fix_pdbs_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --mem=8G

# One-off PDBFixer run on existing PDBs. Submit from pathfinder repo root:
#   sbatch packages/md-runner/sbatch/fix_pdbs_gpu.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source .venv/bin/activate
cd packages/md-runner

python -c "
from pathlib import Path
from openmm import Platform
from openmm.app import ForceField, PDBFile
from pdbfixer import PDBFixer

PDB_DIR = Path('data/md-runner-remd/data/pdbs')
FF = ForceField('amber14-all.xml', 'implicit/obc1.xml')
platform = Platform.getPlatform('CUDA')

for pdb in sorted(PDB_DIR.glob('*.pdb')):
    print(f'Processing {pdb.name}...')
    fixer = PDBFixer(filename=str(pdb), platform=platform)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0, forcefield=FF)
    with open(pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f'  Wrote {pdb}')
print('Done.')
"
