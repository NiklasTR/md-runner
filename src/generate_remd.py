import logging
from pathlib import Path

import hydra
import numpy as np
import openmm
import rootutils
from omegaconf import DictConfig
from openmm import Platform, unit, CustomCentroidBondForce
from openmm.app import ForceField, PDBFile
from openmmtools import multistate, mcmc, states
from openmmtools.cache import global_context_cache


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def geometric_temps(min_temp: float, max_temp: float, n_states: int) -> unit.Quantity:
    """Generate geometrically spaced temperatures between min_temp and max_temp in Kelvin."""
    return unit.Quantity(
        np.geomspace(min_temp.value_in_unit(unit.kelvin), max_temp.value_in_unit(unit.kelvin), n_states),
        unit.kelvin,
    )


def add_com_restraint(
    system, topology, r0=2.0 * unit.nanometer, k=100.0 * unit.kilojoules_per_mole / unit.nanometer**2
):
    """Add flat-bottomed COM restraint between first two protein chains.
    See: https://cbc-univie.github.io/transformato/_modules/transformato/restraints.html#Restraint._add_flatbottom_parameters

    Arguments
    ---------
    system : openmm.System
        OpenMM system to add force to.
    topology : openmm.app.Topology
        Topology containing chain information.
    r0 : openmm.unit.Quantity
        Flat-bottom radius (default: 2.0 nm).
    k : openmm.unit.Quantity
        Spring constant for distances > r0 (default: 100 kJ/mol/nm^2).

    Raises
    ------
    ValueError
        If topology has more than 2 chains.
    """
    chains = list(topology.chains())

    if len(chains) < 2:
        return

    if len(chains) > 2:
        raise ValueError(f"COM restraint only supports 2 chains, found {len(chains)}")

    force = CustomCentroidBondForce(2, "step(distance(g1,g2)-r0) * 0.5*k*(distance(g1,g2)-r0)^2")
    force.addPerBondParameter("r0")
    force.addPerBondParameter("k")

    chain1_atoms = [atom.index for atom in chains[0].atoms()]
    chain2_atoms = [atom.index for atom in chains[1].atoms()]

    force.addGroup(chain1_atoms)
    force.addGroup(chain2_atoms)
    force.addBond([0, 1], [r0, k])

    system.addForce(force)


def get_system(topology, forcefield_files, com_restraint=False):
    forcefield = ForceField(*forcefield_files)
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=openmm.app.CutoffNonPeriodic,
        nonbondedCutoff=2.0 * unit.nanometer,
        constraints=openmm.app.HBonds,
    )
    if com_restraint:
        add_com_restraint(system, topology)
    return system


def setup_platform(cfg):
    platform_properties = {}
    if hasattr(cfg, "platform_properties") and cfg.platform_properties is not None:
        platform_properties = dict(cfg.platform_properties)
        if "Threads" in platform_properties:
            platform_properties["Threads"] = str(platform_properties["Threads"])
    platform = Platform.getPlatform(cfg.platform_name)
    global_context_cache.set_platform(platform, platform_properties)
    logger.info(f"Platform name: {cfg.platform_name} properties: {platform_properties}")
    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_remd.yaml")
def generate_remd(cfg: DictConfig) -> None:  # noqa: C901
    assert cfg.frame_interval > 0
    assert cfg.time_ns > 0
    assert cfg.timestep_fs > 0

    assert cfg.get("pdb_dir") is not None or (
        cfg.get("seq_filename") is not None and cfg.get("seq_idx") is not None
    ), "Either 'pdb_dir' or both 'seq_filename' and 'seq_idx' must be specified in the config"

    if cfg.get("seq_name") is not None:
        pdb_path = Path(cfg.pdb_dir) / f"{cfg.seq_name}.pdb"
    else:
        with Path(cfg.seq_filename).open() as f:
            sequences = f.read().strip().splitlines()
        sequence = sequences[cfg.seq_idx]
        pdb_path = Path(cfg.pdb_dir) / f"{sequence}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found at {pdb_path}")

    setup_platform(cfg)

    pdb = PDBFile(str(pdb_path))
    topology = pdb.getTopology()
    positions = pdb.getPositions(asNumpy=True)

    # Calculate number of frames from time period
    # Each integration step is timestep_fs fs, frame interval steps between frames
    # time_ns * 1e6 fs/ns = total time in fs = num_frames * frame_interval * timestep_fs
    num_frames = int(cfg.time_ns * 1e6 / (cfg.frame_interval * cfg.timestep_fs))

    system = get_system(topology, cfg.forcefield_files, cfg.com_restraint)

    temperatures = geometric_temps(cfg.min_temp * unit.kelvin, cfg.max_temp * unit.kelvin, cfg.n_states)
    logger.info(
        f"Simulating system {pdb_path} with {cfg.n_states} replicas at temperatures: "
        f"{', '.join([f'{t.value_in_unit(unit.kelvin):.1f} K' for t in temperatures])}"
    )
    logger.info(
        f"Total simulation frames per replica to generate: {num_frames} "
        f"calculated from {cfg.time_ns:,} ns / ({cfg.frame_interval:,} * {cfg.timestep_fs} fs per saved frame).",
    )
    thermodynamic_states = [states.ThermodynamicState(system=system, temperature=temp) for temp in temperatures]
    sampler_state = states.SamplerState(
        positions=positions,
        box_vectors=system.getDefaultPeriodicBoxVectors() if system.usesPeriodicBoundaryConditions() else None,
    )

    move = mcmc.LangevinDynamicsMove(
        timestep=cfg.timestep_fs * unit.femtosecond,
        collision_rate=0.3 / unit.picosecond,
        n_steps=cfg.frame_interval,
    )

    sampler = multistate.ReplicaExchangeSampler(
        mcmc_moves=move,
        number_of_iterations=num_frames,
        # Non-reversible parallel tempering (DEO)
        replica_mixing_scheme="swap-neighbors",
        deterministic_swap_order=True,
    )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    nc_path = Path(cfg.output_dir) / "remd.nc"
    ckpt_path = Path(cfg.output_dir) / "remd_checkpoint.nc"

    reporter = multistate.MultiStateReporter(
        nc_path,
        checkpoint_interval=1,  # Save coords and velocities every swap attempt
        checkpoint_storage=ckpt_path,
    )

    if nc_path.exists() and ckpt_path.exists():
        logger.info(f"Resuming from existing simulation files: {nc_path}, {ckpt_path}")
        sampler = multistate.ReplicaExchangeSampler.from_storage(reporter)
        is_minimized = bool(getattr(reporter._storage_checkpoint, "is_minimized", 0))
        is_equilibrated = bool(getattr(reporter._storage_checkpoint, "is_equilibrated", 0))
    else:
        logger.info("Starting new REMD simulation from scratch")
        sampler.create(thermodynamic_states, [sampler_state] * cfg.n_states, reporter)
        reporter._storage_checkpoint.is_minimized = 0
        reporter._storage_checkpoint.is_equilibrated = 0
        reporter.sync()
        is_minimized = False
        is_equilibrated = False

    if not is_equilibrated:
        if not is_minimized:
            sampler.minimize()
            reporter._storage_checkpoint.is_minimized = 1
            reporter.sync()
            logger.info("Minimized, running warmup/equilibration...")
        sampler.equilibrate(cfg.warmup_steps)
        reporter._storage_checkpoint.is_equilibrated = 1
        reporter.sync()
        logger.info(f"Warmup done, {cfg.warmup_steps} steps per replica.")

    sampler.run()


if __name__ == "__main__":
    generate_remd()
