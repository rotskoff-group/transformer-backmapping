from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools import integrators
from mdtraj.reporters import HDF5Reporter
import numpy as np
import math
import mdtraj as md


class Protein:
    """Base class for OpenMM-based protein simulations
    """
    def __init__(self, topology, system, integrator_to_use, platform, positions,
                 do_energy_minimization=True, enforce_periodic_box = True,
                 save_int=10000,
                 save_filename="", chkpt_filename=None,
                 temperature=300.0, dt=0.002, friction=0.1):
        """
        Arguments:
            topology: An OpenMM topology object
            system: An OpenMM system object
            integrator_to_use: A string representing the integrator to use
            platform: An OpenMM platform object
            positions: An OpenMM Quantity object representing the positions of the system with OpenMM units
            do_energy_minimization: A bool representing whether to do energy minimization
            enforce_periodic_box: A bool representing whether to enforce periodic boundary conditions
            save_int: An int representing the frequency for which to save data points to the reporter
            save_filename: A string representing the prefix to add to a file
            chkpt_filename: A string representing the checkpoint file to load
            temperature: A float representing the temperature in K
            dt: A float representing the time step in ps
            friction: A float representing the friction coefficient in ps^-1
        """
        temperature = temperature * kelvin
        friction /= picoseconds
        dt = dt * picoseconds
        self.beta = 1/(temperature * BOLTZMANN_CONSTANT_kB *
                       AVOGADRO_CONSTANT_NA)

        if integrator_to_use == "ovrvo":
            integrator = self._get_ovrvo_integrator(temperature, friction, dt)
        elif integrator_to_use == "verlet":
            integrator = self._get_verlet_integrator(temperature, friction, dt)
        elif integrator_to_use == "omm_ovrvo":
            integrator = integrators.VVVRIntegrator(temperature, friction, dt)
        elif integrator_to_use == "overdamped":
            integrator = self._get_overdamped_integrator(temperature,
                                                         friction, dt)
        elif integrator_to_use == "overdamped_custom_noise":
            integrator = self._get_overdamped_integrator_custom_noise(temperature,
                                                                      friction, dt)
        else:
            raise ValueError("Incorrect integrator supplied")
        self.simulation = Simulation(topology, system, integrator, platform)
        if chkpt_filename is None:
            self.simulation.context.setPositions(positions)
            if do_energy_minimization:
                self.simulation.minimizeEnergy()
            self.simulation.context.setVelocitiesToTemperature(temperature)
        else:
            self.simulation.loadCheckpoint(chkpt_filename)

        self.target_atom_indices = self._get_target_atom_indices()
        all_reporters = [HDF5Reporter(f"{save_filename}.h5", save_int, atomSubset=self.target_atom_indices, enforcePeriodicBox=enforce_periodic_box),
                         CheckpointReporter(f"{save_filename}.chk", save_int)]
        for reporter in all_reporters:
            self.simulation.reporters.append(reporter)
        self.temperature = temperature
        self.save_filename = save_filename
        if integrator_to_use == "ovrvo":
            self.simulation.integrator.setGlobalVariableByName(
                "a", math.exp(-friction * dt/2))
            self.simulation.integrator.setGlobalVariableByName(
                "b", np.sqrt(1 - np.exp(-2 * friction * dt/2)))
            self.simulation.integrator.setGlobalVariableByName(
                "kT",  1/self.beta)

    def _get_overdamped_integrator_custom_noise(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out overdamped Brownian integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            overdamped_integrator: OpenMM Integrator
        """

        overdamped_integrator = CustomIntegrator(dt)
        overdamped_integrator.addGlobalVariable("kT", 1/self.beta)
        overdamped_integrator.addGlobalVariable("friction", friction)
        overdamped_integrator.addPerDofVariable("eta", 0)

        overdamped_integrator.addUpdateContextState()
        overdamped_integrator.addComputePerDof(
            "x", "x+dt*f/(m*friction) + eta*sqrt(2*kT*dt/(m*friction))")
        return overdamped_integrator

    def _get_overdamped_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out overdamped Brownian integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            overdamped_integrator: OpenMM Integrator
        """

        overdamped_integrator = CustomIntegrator(dt)
        overdamped_integrator.addGlobalVariable("kT", 1/self.beta)
        overdamped_integrator.addGlobalVariable("friction", friction)

        overdamped_integrator.addUpdateContextState()
        overdamped_integrator.addComputePerDof(
            "x", "x+dt*f/(m*friction) + gaussian*sqrt(2*kT*dt/(m*friction))")
        return overdamped_integrator

    def _get_verlet_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out Verlet integration
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            verlet_integrator: OpenMM Integrator
        """

        verlet_integrator = CustomIntegrator(dt)
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        verlet_integrator.addComputePerDof("x", "x+dt*v")
        verlet_integrator.addUpdateContextState()
        verlet_integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        return verlet_integrator

    def _get_ovrvo_integrator(self, temperature, friction, dt):
        """Creates OpenMM integrator to carry out ovrvo integration (Sivak, Chodera, and Crooks 2014)
        Arguments:
            temperature: temperature with OpenMM units
            friction: friction coefficient with OpenMM units
            dt: time step with OpenMM units
        Returns:
            ovrvo_integrator: OpenMM Integrator
        """
        ovrvo_integrator = CustomIntegrator(dt)
        ovrvo_integrator.setConstraintTolerance(1e-8)
        ovrvo_integrator.addGlobalVariable("a", math.exp(-friction * dt/2))
        ovrvo_integrator.addGlobalVariable(
            "b", np.sqrt(1 - np.exp(-2 * friction * dt/2)))
        ovrvo_integrator.addGlobalVariable("kT", 1/self.beta)
        ovrvo_integrator.addPerDofVariable("x1", 0)

        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*(f/m)")
        ovrvo_integrator.addConstrainVelocities()

        ovrvo_integrator.addComputePerDof("x", "x + dt*v")
        ovrvo_integrator.addComputePerDof("x1", "x")
        ovrvo_integrator.addConstrainPositions()
        ovrvo_integrator.addComputePerDof("v", "v + (x-x1)/dt")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addUpdateContextState()

        ovrvo_integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
        ovrvo_integrator.addConstrainVelocities()
        ovrvo_integrator.addComputePerDof(
            "v", "(a * v) + (b * sqrt(kT/m) * gaussian)")
        ovrvo_integrator.addConstrainVelocities()
        return ovrvo_integrator


    def _get_target_atom_indices(self):
        """Gets the indices of all non H2O atoms
        Returns:
            all_atom_indices: The indices of all non water atoms
        """
        all_atom_indices = []
        for residue in self.simulation.topology.residues():
            if residue.name != "HOH":
                for atom in residue.atoms():
                    all_atom_indices.append(atom.index)
        return all_atom_indices

    def update_position_and_velocities(self, positions, velocities, use_nanometer_length=False):
        """Updates position and velocities of the system
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in angstroms/ps
            use_nanometer_length: A bool representing whether to use nanometer length in which case the positions 
                                  and velocities are in nanometers and nanometers/picosecond respectively
        """
        if use_nanometer_length:
            length_units = nanometers
        else:
            length_units = angstroms
        positions = positions * length_units
        self.simulation.context.setPositions(positions)
        if velocities is None:
            self.simulation.context.setVelocitiesToTemperature(
                self.temperature)
        else:
            velocities = velocities * (length_units/picosecond)
            self.simulation.context.setVelocities(velocities)

    def relax_energies(self, positions, velocities=None, num_relax_steps=5, use_nanometer_length=True):
        """Carries out num_relax_steps of integration
        Arguments:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in angstroms/ps. If None,
                        velocities are initialized to temperature
            num_relax_steps: Number of time steps of dynamics to run
            use_nanometer_length: A bool representing whether to use nanometer length in which case the positions 
                                  and velocities are in nanometers and nanometers/picosecond respectively
        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms 
                       or nanometers if use_nanometer_length is True
            pe: A float coressponding to the potential energy in kT
            ke: A float coressponding to the kinetic energy in kT
        """
        self.update_position_and_velocities(positions, velocities, use_nanometer_length=use_nanometer_length)
        self.run_sim(num_relax_steps)
        positions, velocities, forces, pe, ke = self.get_information(energies_in_kt=True, 
                                                                     use_nanometer_length=use_nanometer_length)

        positions = positions._value
        return positions, pe, ke

    def run_sim(self, steps, close_file=False):
        """Runs self.simulation for steps steps
        Arguments:
            steps: The number of steps to run the simulation for
            close_file: A bool to determine whether to close file. Potentially necessary
            if using HDF5Reporter
        """
        self.simulation.step(steps)
        if close_file:
            self.simulation.reporters[0].close()

    def get_information(self, as_numpy=True, enforce_periodic_box=True, use_nanometer_length = False, 
                        energies_in_kt = False):
        """Gets information (positions, forces and PE of system)
        Arguments:
            as_numpy: A boolean of whether to return as a numpy array
            enforce_periodic_box: A boolean of whether to enforce periodic boundary conditions
            use_nanometer_length: A bool representing whether to use nanometer length in which case the positions 
                                  and velocities are in nanometers and nanometers/picosecond respectively
            energies_in_kt: A bool representing whether to return energies in kT. If False, energies are returned in kcal/mol

        Returns:
            positions: A numpy array of shape (n_atoms, 3) corresponding to the positions in angstroms or nanometers
                       if use_nanometer_length is True
            velocities: A numpy array of shape (n_atoms, 3) corresponding to the velocities in angstroms/ps or nanometers/picosecond
                        if use_nanometer_length is True
            forces: A numpy array of shape (n_atoms, 3) corresponding to the force in kcal/mol*angstroms or kcal/mol*nanometers
                    if use_nanometer_length is True
            pe: A float coressponding to the potential energy in kcal/mol or kT if energies_in_kt is True
            ke: A float coressponding to the kinetic energy in kcal/mol or kT if energies_in_kt is True
        """
        state = self.simulation.context.getState(getForces=True,
                                                 getEnergy=True,
                                                 getPositions=True,
                                                 getVelocities=True,
                                                 enforcePeriodicBox=enforce_periodic_box)
        if use_nanometer_length:
            length_units = nanometers
        else:
            length_units = angstroms
        positions = state.getPositions(
            asNumpy=as_numpy).in_units_of(length_units)
        forces = state.getForces(asNumpy=as_numpy).in_units_of(
            kilocalories_per_mole / length_units)
        velocities = state.getVelocities(
            asNumpy=as_numpy).in_units_of(length_units / picoseconds)

        positions = positions[self.target_atom_indices]
        forces = forces[self.target_atom_indices]
        velocities = velocities[self.target_atom_indices]

        if energies_in_kt:
            pe = state.getPotentialEnergy() * self.beta
            ke = state.getKineticEnergy() * self.beta
        else:
            pe = state.getPotentialEnergy().in_units_of(kilocalories_per_mole)._value
            ke = state.getKineticEnergy().in_units_of(kilocalories_per_mole)._value

        return positions, velocities, forces, pe, ke

    def generate_long_trajectory(self, num_data_points=int(1E6), save_freq=250, enforce_periodic_box=True):
        """Generates long trajectory of length num_data_points*save_freq time steps where information (pos, vel, forces, pe, ke)
           are saved every save_freq time steps
        Arguments:
            num_data_points: An int representing the number of data points to generate
            save_freq: An int representing the frequency for which to save data points
            tag: A string representing the prefix to add to a file
        Saves:
            tag + "_positions.txt": A text file of shape (num_data_points*n_atoms,3) representing the positions of the trajectory in units of angstroms
            tag + "_velocities.txt": A text file of shape (num_data_points*n_atoms,3) representing the velocities of the trajectory in units of  angstroms/picoseconds
            tag + "_forces.txt": A text file of shape (num_data_points*n_atoms,3) representing the forces of the trajectory in units of kcal/mol*angstroms
            tag + "_pe.txt": A text file of shape (num_data_points,) representing the pe of the trajectory in units of kcal/mol
            tag + "_ke.txt": A text file of shape (num_data_points,) representing the ke of the trajectory in units of kcal/mol
        """
        all_positions = []
        all_velocities = []
        all_forces = []
        all_pe = []
        all_ke = []

        for i in range(num_data_points):
            self.run_sim(save_freq)
            positions, velocities, forces, pe, ke = self.get_information(enforce_periodic_box=enforce_periodic_box)
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_forces.append(forces)
            all_pe.append(pe)
            all_ke.append(ke)

            if (i % 10 == 0 or i == (num_data_points - 1)):
                all_positions = np.stack(all_positions)
                f = open(self.save_filename + "_positions.txt", 'ab')
                np.savetxt(f, all_positions.reshape(
                    (-1, all_positions.shape[-1])))
                f.close()

                all_velocities = np.stack(all_velocities)
                f = open(self.save_filename + "_velocities.txt", 'ab')
                np.savetxt(f, all_velocities.reshape(
                    (-1, all_velocities.shape[-1])))
                f.close()

                all_forces = np.stack(all_forces)
                f = open(self.save_filename + "_forces.txt", 'ab')
                np.savetxt(f, all_forces.reshape((-1, all_forces.shape[-1])))
                f.close()

                all_pe = np.stack(all_pe)
                f = open(self.save_filename + "_pe.txt", 'ab')
                np.savetxt(f, all_pe.reshape((-1, 1)))
                f.close()

                all_ke = np.stack(all_ke)
                f = open(self.save_filename + "_ke.txt", 'ab')
                np.savetxt(f, all_ke.reshape((-1, 1)))
                f.close()

                all_positions = []
                all_velocities = []
                all_forces = []
                all_pe = []
                all_ke = []
        chk_reporter = CheckpointReporter(f"{self.save_filename}.chk", 1)
        self.simulation.reporters.append(chk_reporter)
        self.run_sim(1)

class ProteinImplicit(Protein):
    """A class for OpenMM-based protein simulations in implicit solvent
    """
    def __init__(self, filename, chk, integrator_to_use="ovrvo",
                 temperature=300.0, dt=0.002, friction=0.1, save_int=10000,
                 save_filename=None):
        """Arguments:
            filename: A string representing the prefix of the prmtop and crd files
            chk: An int representing the checkpoint number
            integrator_to_use: A string representing the integrator to use
            temperature: A float representing the temperature in K
            dt: A float representing the time step in ps
            friction: A float representing the friction coefficient in ps^-1
            save_int: An int representing the frequency for which to save data points to the reporter
            save_filename: A string representing the prefix to add to a file
        """
        prmtop = AmberPrmtopFile(f'{filename}.prmtop')
        system = prmtop.createSystem(
            implicitSolvent=OBC2,
            implicitSolventKappa=0.1/nanometer,
            constraints=None,
            nonbondedCutoff=None,
            rigidWater=True,
            hydrogenMass=None
        )
        topology = prmtop.topology
        inpcrd = AmberInpcrdFile(f'{filename}.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        platform = Platform.getPlatformByName("CPU")
        if save_filename is None:
            save_filename = filename
        if chk == 0:
            chkpt_filename = None
        else:
            chkpt_filename = f"{save_filename}_{chk - 1}.chk"
        save_filename = f"{save_filename}_{chk}"

        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                         platform=platform,
                         positions=positions, temperature=temperature, dt=dt,
                         friction=friction, chkpt_filename=chkpt_filename,
                         save_filename=save_filename, save_int=save_int)

class ProteinSolvent(Protein):
    """A class for OpenMM-based protein simulations in explicit solvent
    """
    def __init__(self, filename, chk, integrator_to_use="ovrvo",
                 temperature=300.0, dt=0.002, friction=0.1,
                 nonbonded_cutoff=0.9,  ewald_tolerance=1.0E-5, switch_width=0.16,
                 save_int=10000, use_hbond_constraints=True, save_filename=None):
        """Arguments:
            filename: A string representing the prefix of the prmtop and crd files
            chk: An int representing the checkpoint number
            integrator_to_use: A string representing the integrator to use
            temperature: A float representing the temperature in K
            dt: A float representing the time step in ps
            friction: A float representing the friction coefficient in ps^-1
            nonbonded_cutoff: A float representing the nonbonded cutoff in nm
            ewald_tolerance: A float representing the Ewald tolerance
            switch_width: A float representing the switch width in nm
            save_int: An int representing the frequency for which to save data points to the reporter
            use_hbond_constraints: A bool representing whether to use hydrogen bond constraints
            save_filename: A string representing the prefix to add to a file
        """
        nonbonded_cutoff = nonbonded_cutoff * nanometer
        switch_width = switch_width * nanometer
        prmtop = AmberPrmtopFile(f'{filename}.prmtop')

        if use_hbond_constraints:
            constraints = HBonds
        else:
            constraints = None
        system = prmtop.createSystem(constraints=constraints, nonbondedMethod=PME,
                                     nonbondedCutoff=nonbonded_cutoff, rigidWater=True,
                                     hydrogenMass=None,)
        topology = prmtop.topology
        system.addForce(MonteCarloBarostat(1*bar, temperature, 25))
        forces = {system.getForce(index).__class__.__name__: system.getForce(index)
                  for index in range(system.getNumForces())}
        forces['NonbondedForce'].setUseDispersionCorrection(True)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewald_tolerance)
        forces['NonbondedForce'].setUseSwitchingFunction(True)
        forces['NonbondedForce'].setSwitchingDistance((nonbonded_cutoff
                                                       - switch_width))
        inpcrd = AmberInpcrdFile(f'{filename}.crd')
        positions = inpcrd.getPositions(asNumpy=True)
        box_vectors = inpcrd.getBoxVectors(asNumpy=True)
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        platform = Platform.getPlatformByName("CUDA")
        if save_filename is None:
            save_filename = filename
        if chk == 0:
            chkpt_filename = None
        else:
            chkpt_filename = f"{save_filename}_{chk - 1}.chk"
        save_filename = f"{save_filename}_{chk}"

        super().__init__(topology=topology, system=system, integrator_to_use=integrator_to_use,
                         platform=platform,
                         positions=positions, temperature=temperature, dt=dt,
                         friction=friction, chkpt_filename=chkpt_filename,
                         save_filename=save_filename, save_int=save_int)

