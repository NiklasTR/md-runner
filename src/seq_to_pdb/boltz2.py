"""Boltz-2-based sequence to PDB conversion. Uses single-sequence mode (msa: empty)."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import hydra
import openmm.app
import rootutils
from omegaconf import DictConfig
from openmm.app import ForceField
from pdbfixer import PDBFixer
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

_AMBER14_FORCEFIELD = ForceField("amber14-all.xml", "implicit/obc1.xml")

_YAML_TEMPLATE = """sequences:
  - protein:
      id: A
      sequence: {sequence}
      msa: empty
"""


def _fix_pdb_with_pdbfixer(pdb_path: Path, out_path: Path, *, pH: float = 7.0) -> None:
    """Add hydrogens and fix protonation with PDBFixer."""
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH, forcefield=_AMBER14_FORCEFIELD)
    with open(out_path, "w") as f:
        openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, f)


def make_peptide_with_boltz(
    sequence: str,
    save_path: Path,
    *,
    accelerator: str = "gpu",
    fix_with_pdbfixer: bool = True,
    pH: float = 7.0,
) -> None:
    """Generate a PDB file for a peptide sequence using Boltz-2 (single-sequence, no MSA).

    Args:
        sequence: One-letter amino acid sequence (e.g. "ACD").
        save_path: Path where the generated PDB file will be saved.
        accelerator: "gpu" or "cpu".

    Raises:
        RuntimeError: If boltz predict fails.
        FileNotFoundError: If Boltz does not produce the expected output file.
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = _YAML_TEMPLATE.format(sequence=sequence)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        yaml_path = tmpdir_path / "input.yaml"
        yaml_path.write_text(yaml_content)

        out_dir = tmpdir_path / "out"
        out_dir.mkdir()

        result = subprocess.run(
            [
                "boltz",
                "predict",
                str(yaml_path),
                "--out_dir",
                str(out_dir),
                "--output_format",
                "pdb",
                "--accelerator",
                accelerator,
            ],
            cwd=tmpdir_path,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            stderr_preview = (result.stderr or "").strip().splitlines()
            stderr_preview = "\n".join(stderr_preview[-20:])
            raise RuntimeError(
                f"boltz predict failed with exit code {result.returncode}.\nstderr (last 20 lines):\n{stderr_preview}",
            )

        pred_files = list(out_dir.rglob("*_model_0.pdb"))

        if not pred_files:
            raise FileNotFoundError(
                f"No output PDB produced by Boltz in {out_dir}",
            )

        boltz_pdb = pred_files[0]
        if fix_with_pdbfixer:
            _fix_pdb_with_pdbfixer(boltz_pdb, save_path, pH=pH)
        else:
            shutil.copy(boltz_pdb, save_path)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="seq_to_pdb.yaml")
def seq_to_pdb(cfg: DictConfig) -> None:
    """Convert amino acid sequences to PDB files using Boltz-2 (single-sequence, no MSA).

    Reads sequences from seq_filename or seq_name, then generates PDB files for each.
    """
    pdb_dir = Path(cfg.paths.data_dir) / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.seq_filename is not None or cfg.seq_name is not None, (
        "Either seq_filename or seq_name must be provided",
    )
    assert cfg.seq_filename is None or cfg.seq_name is None, (
        "Only one of seq_filename or seq_name must be provided",
    )

    if cfg.seq_filename is not None:
        with Path(cfg.seq_filename).open() as f:
            sequences = [line.strip() for line in f if line.strip()]
    else:
        sequences = [cfg.seq_name]

    accelerator = getattr(cfg, "accelerator", "gpu")
    fix_with_pdbfixer = getattr(cfg, "fix_with_pdbfixer", True)
    pH = getattr(cfg, "pH", 7.0)

    for sequence in tqdm(sequences):
        save_path = pdb_dir / f"{sequence}.pdb"
        make_peptide_with_boltz(
            sequence,
            save_path,
            accelerator=accelerator,
            fix_with_pdbfixer=fix_with_pdbfixer,
            pH=pH,
        )


if __name__ == "__main__":
    seq_to_pdb()
