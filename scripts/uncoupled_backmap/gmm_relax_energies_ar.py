import torch
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
from cgp2 import ProteinImplicit, SideChainLens
import os

from openmm import *
from openmm.app import *
from openmm.unit import *
import argparse

device = torch.device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--bb_index", type=int, default=0)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--num_steps", type=int, default=1)
parser.add_argument("--prop_temp", type=float, default=1)


config = parser.parse_args()
bb_index = config.bb_index
dt = config.dt
num_steps = config.num_steps
prop_temp = config.prop_temp

traj_folder_name = "/scratch/users/shriramc/datasets/ar_robustelli/fg_traj/"
# all_subsampled_indices = np.load(traj_folder_name + "all_subsampled_indices.npy")
all_subsampled_indices = np.arange(57144)


protein_filename = f"{traj_folder_name}/ar_traj.h5"
protein_traj = md.load(protein_filename, frame=0)
protein_top = protein_traj.topology

side_chain_lens = SideChainLens(protein_top=protein_top)
side_chain_info = side_chain_lens.get_data(protein_traj)
cg_indices = side_chain_lens.atom_indices_coarser
gen_indices = np.array([index for index in side_chain_lens.atom_indices_finer if index not in cg_indices])

p = ProteinImplicit(filename=f"{traj_folder_name}/ar", 
                    integrator_to_use = "overdamped", 
                    save_filename=f"init_energies_{bb_index}_dt_{dt}_num_steps_{num_steps}_prop_temp_{prop_temp}", 
                    chk=0, dt=dt, friction=100.0)
for index in cg_indices:
    p.simulation.system.setParticleMass(index, 0.0)



# bb_config = all_subsampled_indices[bb_index]




T = prop_temp * kelvin
all_scales = np.array([((AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB * T)/(p.simulation.system.getParticleMass(index).in_units_of(kilograms/mole))).sqrt().in_units_of(nanometers/picosecond)._value
                       for index in gen_indices])
all_scales = np.repeat(all_scales, 3)

all_scales = torch.tensor(all_scales, device=device).float()

velocity_sampler = torch.distributions.normal.Normal(loc=torch.zeros_like(all_scales), scale=all_scales)

for bb_config in all_subsampled_indices[bb_index * 15000:(bb_index + 1) * 15000]:
    save_folder_name = f"./overdamped_prop_temp_{prop_temp}_dt_{dt}_num_steps_{num_steps}/{bb_config}/"
    os.makedirs(save_folder_name, exist_ok=True)
    root_load_folder_name = f"./mcmc_dataset/{bb_config}/"

    for batch_index in range(1):
        try:
            reconstructed_traj = md.load(f"{root_load_folder_name}/mcmc_{batch_index}.h5")
            original_energies = np.load(f"{root_load_folder_name}/mcmc_energies_{batch_index}.npy")
        except FileNotFoundError:
            continue
        indices_to_keep = np.where((~np.isnan(original_energies) 
                                    & ~np.isinf(original_energies)
                                    & (original_energies < 1E5)))[0]
        all_reconstructed_positions = reconstructed_traj.xyz
        all_reconstructed_positions = all_reconstructed_positions[indices_to_keep]
        num_samples = all_reconstructed_positions.shape[0]


        all_velocities_gen = velocity_sampler.sample((num_samples,)).to(device)
        all_velocities_log_prob = velocity_sampler.log_prob(all_velocities_gen).sum(-1)
        all_velocities_gen = all_velocities_gen.reshape(-1, all_scales.shape[0]//3, 3)

        all_velocities = torch.zeros((num_samples, p.simulation.system.getNumParticles(), 3), device=device)
        all_velocities[:, gen_indices, :] = all_velocities_gen
        all_velocities = all_velocities.detach().cpu().numpy()
        all_velocities_log_prob = all_velocities_log_prob.detach().cpu().numpy()

        np.save(f"{save_folder_name}/all_velocities_{batch_index}.npy", all_velocities)
        np.save(f"{save_folder_name}/all_velocities_log_prob_{batch_index}.npy", all_velocities_log_prob)
        np.save(f"{save_folder_name}/indices_to_keep_{batch_index}.npy", indices_to_keep)




        all_relaxed_positions = []
        all_relaxed_potential_energies = []
        all_relaxed_kinetic_energies = []
        for (positions, velocities) in zip(all_reconstructed_positions, all_velocities):
            try:
                relaxed_positions, relaxed_potential_energy, relaxed_kinetic_energy = p.relax_energies(positions=positions, velocities=velocities,
                                                                                                    num_relax_steps=num_steps, use_nanometer_length=True)
                relaxed_positions, relaxed_potential_energy, relaxed_kinetic_energy
            except (OpenMMException, ValueError):
                relaxed_positions = np.zeros_like(positions)
                relaxed_potential_energy = np.inf
                relaxed_kinetic_energy = np.inf
            all_relaxed_positions.append(relaxed_positions)
            all_relaxed_potential_energies.append(relaxed_potential_energy)
            all_relaxed_kinetic_energies.append(relaxed_kinetic_energy)
        all_relaxed_positions = np.stack(all_relaxed_positions)
        all_relaxed_potential_energies = np.array(all_relaxed_potential_energies)
        all_relaxed_kinetic_energies = np.array(all_relaxed_kinetic_energies)
        np.save(f"{save_folder_name}/all_relaxed_positions_{batch_index}.npy", all_relaxed_positions)
        np.save(f"{save_folder_name}/all_relaxed_potential_energies_{batch_index}.npy", all_relaxed_potential_energies)
        np.save(f"{save_folder_name}/all_relaxed_kinetic_energies_{batch_index}.npy", all_relaxed_kinetic_energies)

