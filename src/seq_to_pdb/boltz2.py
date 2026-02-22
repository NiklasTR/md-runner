"""Boltz-2-based sequence to PDB conversion. Uses single-sequence mode (msa: empty).

Error location: The cyclic peptide failure occurs during OpenMM parameterisation
(ForceField.createSystem), called from within PDBFixer.addMissingHydrogens ->
Modeller.addHydrogens. The first residue is matched to NPRO (N-terminal template)
because the topology lacks the head-to-tail bond; NPRO expects an external C atom
from a "previous" residue that does not exist in a cyclic chain.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import hydra
import openmm.app
import openmm.unit
from omegaconf import DictConfig
from openmm.app import ForceField, Modeller, PDBFile
from pdbfixer import PDBFixer
from tqdm import tqdm

from src.utils.sequence_parser import d_positions_and_ccds, line_to_name, parse_sequence_line
from src.utils.topology_io import load_npz, save_npz, write_xyz

logger = logging.getLogger(__name__)
_AMBER14_FORCEFIELD = ForceField("amber14-all.xml", "implicit/obc1.xml")

class _CyclicForceField(ForceField):
    """ForceField that passes ignoreExternalBonds=True for cyclic peptide template matching."""

    def createSystem(
        self,
        topology,
        nonbondedMethod=openmm.app.NoCutoff,
        nonbondedCutoff=1.0 * openmm.unit.nanometer,
        constraints=None,
        rigidWater=None,
        removeCMMotion=True,
        hydrogenMass=None,
        residueTemplates=dict(),
        ignoreExternalBonds=False,
        switchDistance=None,
        flexibleConstraints=False,
        drudeMass=0.4 * openmm.unit.amu,
        **args,
    ):
        return super().createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=constraints,
            rigidWater=rigidWater,
            removeCMMotion=removeCMMotion,
            hydrogenMass=hydrogenMass,
            residueTemplates=residueTemplates,
            ignoreExternalBonds=True,
            switchDistance=switchDistance,
            flexibleConstraints=flexibleConstraints,
            drudeMass=drudeMass,
            **args,
        )


def _build_boltz_yaml(
    sequence: str, cyclic: bool, modifications: list[tuple[int, str]] | None = None
) -> str:
    """Build Boltz YAML. Sequence uses lowercase for D-amino acids."""
    seq_upper = sequence.upper()
    mods = modifications if modifications is not None else d_positions_and_ccds(sequence)
    mods_block = ""
    if mods:
        mods_lines = [
            f"        - position: {pos}\n          ccd: {ccd}"
            for pos, ccd in mods
        ]
        mods_block = "\n      modifications:\n" + "\n".join(mods_lines)
    return f"""sequences:
  - protein:
      id: A
      sequence: {seq_upper}
      msa: empty
      cyclic: {str(cyclic).lower()}{mods_block}
"""


_D_TO_L: dict[str, str] = {
    "DAL": "ALA", "DAR": "ARG", "DSG": "ASN", "DAS": "ASP",
    "DCY": "CYS", "DGL": "GLU", "DGN": "GLN", "DHI": "HIS",
    "DIL": "ILE", "DLE": "LEU", "DLY": "LYS", "MED": "MET",
    "DPN": "PHE", "DPR": "PRO", "DSN": "SER", "DTH": "THR",
    "DTR": "TRP", "DTY": "TYR", "DVA": "VAL",
}


def _internal_hydrogen_variant(res_name: str, pH: float) -> list[tuple[str, str]] | None:
    """Build an explicit hydrogen list treating the residue as internal (not terminal).

    For D-amino acids, uses the L counterpart's hydrogen definitions (same bonded
    parameters per ff14SB). For HIS, picks HID (neutral) at pH >= 6 or HIP at pH < 6.
    """
    l_name = _D_TO_L.get(res_name, res_name)
    Modeller._loadStandardHydrogenDefinitions()
    spec = Modeller._residueHydrogens.get(l_name)
    if spec is None:
        return None
    variant = None
    if l_name == "HIS":
        variant = "HIP" if pH < 6.0 else "HID"
    return [
        (h.name, h.parent) for h in spec.hydrogens
        if pH <= h.maxph
        and (h.terminal is None or "-" in h.terminal)
        and (h.variants is None or variant is None or variant in h.variants)
    ]


def _add_chain_bonds(topology: openmm.app.Topology, *, cyclic: bool = False) -> None:
    """Add all bonds between residues in each chain.

    Adds C(i)-N(i+1) for consecutive residues. For cyclic peptides, also adds
    N(first)-C(last) to close the ring. Skips bonds that already exist.
    """
    existing = {frozenset((b[0].index, b[1].index)) for b in topology.bonds()}
    for chain in topology.chains():
        residues = list(chain.residues())
        for i in range(len(residues) - 1):
            c_atom = next((a for a in residues[i].atoms() if a.name == "C"), None)
            n_atom = next((a for a in residues[i + 1].atoms() if a.name == "N"), None)
            if c_atom is not None and n_atom is not None:
                pair = frozenset((c_atom.index, n_atom.index))
                if pair not in existing:
                    topology.addBond(c_atom, n_atom)
                    existing.add(pair)
        if cyclic and len(residues) >= 2:
            n_atom = next((a for a in residues[0].atoms() if a.name == "N"), None)
            c_atom = next((a for a in residues[-1].atoms() if a.name == "C"), None)
            if n_atom is not None and c_atom is not None:
                pair = frozenset((n_atom.index, c_atom.index))
                if pair not in existing:
                    topology.addBond(n_atom, c_atom)


def _fix_pdb(
    pdb_path: Path,
    *,
    pH: float = 7.0,
    cyclic: bool = False,
    debug_dir: Path | None = None,
) -> tuple[openmm.app.Topology, list]:
    """Add missing atoms/hydrogens, fix protonation. Returns (topology, positions) in memory."""
    fixer = PDBFixer(filename=str(pdb_path))
    # TODO: findMissingResidues/findMissingAtoms/addMissingAtoms may add terminal atoms
    # (e.g. OXT) that are wrong for cyclic peptides. Consider dropping PDBFixer for this
    # step entirely and handling missing heavy atoms via Modeller directly.
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if cyclic:
        oxt_atoms = [
            a for r in fixer.topology.residues() for a in r.atoms() if a.name == "OXT"
        ]
        if oxt_atoms:
            modeller = Modeller(fixer.topology, fixer.positions)
            modeller.delete(oxt_atoms)
            fixer.topology = modeller.topology
            fixer.positions = modeller.positions

    def _save_debug(suffix: str, topo: openmm.app.Topology, pos) -> None:
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            p = debug_dir / f"{pdb_path.stem}_before_hydrogens_{suffix}.pdb"
            with open(p, "w") as f:
                PDBFile.writeFile(topo, pos, f)
            logger.warning("Saved interim topology to %s", p)

    try:
        if cyclic:
            _add_chain_bonds(fixer.topology, cyclic=True)
            residues = list(fixer.topology.residues())
            variants = [None] * len(residues)
            for i, res in enumerate(residues):
                is_d = res.name in _D_TO_L
                is_terminal = (i == 0 or i == len(residues) - 1)
                if is_d or is_terminal:
                    variants[res.index] = _internal_hydrogen_variant(res.name, pH)
            modeller = Modeller(fixer.topology, fixer.positions)
            modeller.addHydrogens(
                pH=pH,
                forcefield=_CyclicForceField("amber14-all.xml", "implicit/obc1.xml"),
                variants=variants,
            )
            return modeller.topology, modeller.positions
        else:
            fixer.addMissingHydrogens(pH=pH, forcefield=_AMBER14_FORCEFIELD)
            return fixer.topology, fixer.positions
    except Exception:
        _save_debug("cyclic" if cyclic else "linear", fixer.topology, fixer.positions)
        raise


def _minimize(
    topology: openmm.app.Topology,
    positions,
    *,
    cyclic: bool = False,
) -> tuple[openmm.app.Simulation, openmm.openmm.System]:
    """Energy-minimize in memory. Returns (Simulation, System) after minimization."""
    ff = (
        _CyclicForceField("amber14-all.xml", "implicit/obc1.xml")
        if cyclic
        else _AMBER14_FORCEFIELD
    )
    system = ff.createSystem(
        topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        constraints=openmm.app.HBonds,
    )
    integrator = openmm.openmm.LangevinMiddleIntegrator(
        300 * openmm.unit.kelvin,
        1.0 / openmm.unit.picosecond,
        0.002 * openmm.unit.picosecond,
    )
    simulation = openmm.app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    e_before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy()
    e_after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info("Minimized: %s -> %s", e_before, e_after)
    return simulation, system


def make_peptide_with_boltz(
    sequence: str,
    save_path: Path,
    *,
    accelerator: str = "gpu",
    fix_with_pdbfixer: bool = True,
    pH: float = 7.0,
    cyclic: bool = False,
    modifications: list[tuple[int, str]] | None = None,
) -> None:
    """Generate an NPZ file for a peptide sequence using Boltz-2 (single-sequence, no MSA).

    The NPZ contains: atoms, positions, sequence, topology, system, state.

    Args:
        sequence: One-letter amino acid sequence (e.g. "ACD"). Lowercase for D-amino acids.
        save_path: Path where the generated .npz file will be saved.
        accelerator: "gpu" or "cpu".
        cyclic: If True, set cyclic: true in Boltz config for head-to-tail lactam peptides.

    Raises:
        RuntimeError: If boltz predict fails.
        FileNotFoundError: If Boltz does not produce the expected output file.
    """
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = _build_boltz_yaml(sequence, cyclic, modifications)

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
        debug_dir = save_path.parent / "debug"
        if fix_with_pdbfixer:
            try:
                topology, positions = _fix_pdb(
                    boltz_pdb,
                    pH=pH,
                    cyclic=cyclic,
                    debug_dir=debug_dir,
                )
            except Exception:
                debug_dir.mkdir(parents=True, exist_ok=True)
                raw_debug = debug_dir / f"{save_path.stem}_boltz_raw.pdb"
                shutil.copy(boltz_pdb, raw_debug)
                logger.warning("PDBFixer failed. Saved raw Boltz output to %s", raw_debug)
                raise
        else:
            pdb = PDBFile(str(boltz_pdb))
            topology, positions = pdb.topology, pdb.positions

    simulation, system = _minimize(topology, positions, cyclic=cyclic)

    save_npz(
        str(save_path),
        topology,
        simulation,
        system,
        sequence=sequence,
        is_cyclic=cyclic,
    )

    xyz_path = save_path.with_suffix(".xyz")
    write_xyz(str(xyz_path), load_npz(str(save_path)))
    logger.info("Saved %s and %s", save_path.name, xyz_path.name)


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
            sequences = []
            for line in f:
                parsed = parse_sequence_line(line)
                if parsed is not None:
                    sequences.append((*parsed, line_to_name(line)))
    else:
        raw = cfg.seq_name
        parsed = parse_sequence_line(raw)
        if parsed is None:
            raise ValueError(f"Could not parse sequence: {raw!r}")
        sequences = [(*parsed, line_to_name(raw))]

    accelerator = getattr(cfg, "accelerator", "gpu")
    fix_with_pdbfixer = getattr(cfg, "fix_with_pdbfixer", True)
    pH = getattr(cfg, "pH", 7.0)

    for sequence, is_cyclic, modifications, name in tqdm(sequences):
        save_path = pdb_dir / f"{name}.npz"
        make_peptide_with_boltz(
            sequence,
            save_path,
            accelerator=accelerator,
            fix_with_pdbfixer=fix_with_pdbfixer,
            pH=pH,
            cyclic=is_cyclic,
            modifications=modifications if modifications else None,
        )


if __name__ == "__main__":
    seq_to_pdb()
