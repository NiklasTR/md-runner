"""Lossless serialization of OpenMM Topology + positions + System + State to a single NPZ.

Keys: atoms, positions, sequence, topology, system, state.
Topology is stored as a PDBx/mmCIF string (OpenMM's native topology format).
System and State are stored as OpenMM XML strings.
"""

from io import StringIO

import numpy as np
import openmm
import openmm.app
import openmm.unit


def save_npz(
    path: str,
    topology: openmm.app.Topology,
    simulation: openmm.app.Simulation,
    system: openmm.openmm.System,
    *,
    sequence: str = "",
    is_cyclic: bool = False,
) -> None:
    """Save everything into a single compressed NPZ with 6 keys."""
    atom_list = list(topology.atoms())

    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getParameters=True,
    )
    pos = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)

    cif_buf = StringIO()
    openmm.app.PDBxFile.writeFile(topology, state.getPositions(), cif_buf)

    np.savez_compressed(
        path,
        atoms=np.array([a.element.atomic_number for a in atom_list], dtype=np.int8),
        positions=np.asarray(pos, dtype=np.float64),
        sequence=np.array(sequence),
        topology=np.array(cif_buf.getvalue()),
        system=np.array(openmm.XmlSerializer.serialize(system)),
        state=np.array(openmm.XmlSerializer.serialize(state)),
    )


def load_npz(path: str) -> dict:
    """Load an NPZ and return its contents as a plain dict."""
    return dict(np.load(path, allow_pickle=False))


def topology_from_npz(data: dict) -> openmm.app.Topology:
    """Reconstruct an OpenMM Topology from the PDBx/mmCIF string in the NPZ."""
    cif = openmm.app.PDBxFile(StringIO(str(data["topology"])))
    return cif.topology


def system_from_npz(data: dict) -> openmm.openmm.System:
    """Deserialize an OpenMM System from the NPZ."""
    return openmm.XmlSerializer.deserialize(str(data["system"]))


def positions_from_npz(data: dict) -> openmm.unit.Quantity:
    """Return positions as an OpenMM Quantity in nanometers."""
    return openmm.unit.Quantity(data["positions"], openmm.unit.nanometer)


_ATOMIC_NUMBER_TO_SYMBOL: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S", 17: "Cl",
    9: "F", 35: "Br", 53: "I", 11: "Na", 12: "Mg", 19: "K", 20: "Ca",
    26: "Fe", 29: "Cu", 30: "Zn", 34: "Se",
}


def write_xyz(path: str, data: dict) -> None:
    """Write an XYZ file from NPZ data. Positions are converted from nm to Angstrom."""
    atomic_numbers = data["atoms"]
    positions_nm = data["positions"]
    positions_ang = positions_nm * 10.0
    n_atoms = len(atomic_numbers)

    with open(path, "w") as f:
        f.write(f"{n_atoms}\n")
        f.write(f"{str(data.get('sequence', ''))}\n")
        for z, (x, y, z_coord) in zip(atomic_numbers, positions_ang):
            sym = _ATOMIC_NUMBER_TO_SYMBOL.get(int(z), "X")
            f.write(f"{sym} {x:.6f} {y:.6f} {z_coord:.6f}\n")
