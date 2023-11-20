import torch
import numpy as np
import os
from cgp import ProteinImplicit, SideChainLens, GaussianMixtureModel, ConcatenatedModel, Factory
import mdtraj as md
import argparse


device = torch.device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--bb_index", type=int, default=0)

config = parser.parse_args()
bb_index = config.bb_index

root_protein_folder_name = "./datasets/chignolin/"
all_subsampled_indices = np.arange(50000)

# root_protein_folder_name = "./datasets/ar/"
# all_subsampled_indices = np.arange(57144)

traj_folder_name = f"{root_protein_folder_name}/fg_traj/"

protein_filename = traj_folder_name + "chignolin_traj.h5"
# protein_filename = traj_folder_name + "ar_traj.h5"

protein_traj = md.load(protein_filename, frame=0)
protein_top = protein_traj.topology

p = ProteinImplicit(filename=f"{traj_folder_name}/chignolin",
                    save_filename=f"cg_{bb_index}", chk=0)

# p = ProteinImplicit(filename=f"{traj_folder_name}/ar",
                    # save_filename=f"cg_{bb_index}", chk=0)


network_folder_name = f"{root_protein_folder_name}/gmm/"


def get_gmm_model(in_dim, all_coordinates, num_components, protein_name):

    num_datapoints = 500000
    n_dim = in_dim
    means = torch.randn(num_components, n_dim).to(device)
    all_offset = torch.randn(1, n_dim).to(device).cpu().numpy()
    covs = torch.eye(n_dim).unsqueeze(0).repeat(num_components,
                                                1, 1).to(device)

    weights = torch.ones(num_components, device=device) / num_components

    tag = (str(protein_name) + "_protein_name_"
           + "_".join(all_coordinates) + "_coordinate_"
           + str(num_components) + "_num_components_"
           + str(num_datapoints) + "_num_datapoints")

    folder_name = network_folder_name + tag + "/"

    if "_".join(all_coordinates) == "dihedrals":
       sample_more_info = True
    else:
       sample_more_info = False

    gmm_model = GaussianMixtureModel(means=means, covs=covs, weights=weights,
                                     all_offset=all_offset,
                                     folder_name=folder_name, 
                                     sample_more_info=sample_more_info,
                                     device=device)
    loaded_epoch = gmm_model.load(epoch_num=None)
    print("Loaded epoch: ", loaded_epoch)
    sk_info = np.load(folder_name + "sk_info.npy", allow_pickle=True)

    return gmm_model


all_tpp_names = np.loadtxt(
    f"{root_protein_folder_name}/tpp/sidechain_info/dihedral_name_n_components.txt", dtype=str)


side_chain_lens = SideChainLens(protein_top=protein_top)
side_chain_info = side_chain_lens.get_data(protein_traj)


all_chi_info_by_residue_num = side_chain_info["chi"]

ic_models_by_res_num = {}
cg_c_info_by_res_num = {}
z_matrix_by_res_num = {}
full_z_matrix = []
for (res_num, chi_info) in all_chi_info_by_residue_num.items():
    # if chi_info is None:
    if chi_info is None:
        continue

    if res_num == 0:
        assert protein_top.residue(res_num).name == "ACE"
        protein_name = "ACE"

    elif res_num == protein_top.n_residues - 1:
        assert protein_top.residue(res_num).name == "NME"
        protein_name = "NME"
    elif res_num == 1:
        protein_name = f"{protein_top.residue(res_num).name}_{protein_top.residue(res_num + 1).name}_{protein_top.residue(res_num + 2).name}_FIRST"
    elif res_num == protein_top.n_residues - 2:
        protein_name = f"{protein_top.residue(res_num - 2).name}_{protein_top.residue(res_num - 1).name}_{protein_top.residue(res_num).name}_LAST"
    else:
        protein_name = f"{protein_top.residue(res_num - 1).name}_{protein_top.residue(res_num).name}_{protein_top.residue(res_num + 1).name}"

    in_dim = np.load(
        f"{root_protein_folder_name}/tpp/sidechain_info/{protein_name}/dihedrals.npy").shape[1]
    n_components_dihedrals = all_tpp_names[all_tpp_names[:, 0]
                                           == protein_name, 1].astype(int)[0]

    bond_length_angle_model = get_gmm_model(in_dim * 2, ["distances", "angles"],
                                            1, protein_name)
    dihedral_model = get_gmm_model(in_dim, ["dihedrals"],
                                   n_components_dihedrals, protein_name)

    ic_model = ConcatenatedModel([bond_length_angle_model, dihedral_model])

    ic_models_by_res_num[res_num] = ic_model
    cg_c_info_by_res_num[res_num] = [None, None]
    z_matrix_by_res_num[res_num] = chi_info["dihedral_indices"]
    full_z_matrix.append(chi_info["dihedral_indices"])

full_z_matrix = np.concatenate(full_z_matrix)


side_chain_factory = Factory(lens=side_chain_lens)

for config_index in all_subsampled_indices[bb_index * 50:(bb_index + 1) * 50]:
    uncoupled_gmm_folder_name = f"uncoupled_gmm_dataset/{config_index}/"
    os.makedirs(uncoupled_gmm_folder_name, exist_ok=True)
    protein_traj = md.load(protein_filename, frame=config_index)
    protein_top = protein_traj.topology
    protein_traj = md.Trajectory(protein_traj.xyz.repeat(10000, axis=0),
                                 protein_top)
    side_chain_lens = SideChainLens(protein_top=protein_top)
    cg_traj = protein_traj.atom_slice(side_chain_lens.atom_indices_coarser)

    for sample_num in range(1):
        with torch.no_grad():
            cg_traj_pos = torch.tensor(cg_traj.xyz, device=device).float()
            all_reconstructed_positions, total_log_prob, all_gen_components = side_chain_factory.serial_reconstruct_from_cg_traj(cg_traj_pos=cg_traj_pos,
                                                                                                                                 ic_models_by_res_num=ic_models_by_res_num,
                                                                                                                                 cg_c_info_by_res_num=cg_c_info_by_res_num,
                                                                                                                                 grad_enabled=False,
                                                                                                                                 z_matrix_by_res_num=z_matrix_by_res_num,
                                                                                                                                 return_gen_components=True,
                                                                                                                                 device=device)
            rec_traj = md.Trajectory(all_reconstructed_positions.cpu().numpy(),
                                     protein_top)

        all_reconstructed_positions = rec_traj.xyz
        all_indices_to_keep = []
        all_energies = []
        for (pos_index, positions) in enumerate(all_reconstructed_positions):
            p.update_position_and_velocities(positions=positions, 
                                             velocities=None, use_nanometer_length=True)
            _, _, _, pe, _ = p.get_information(as_numpy=True, 
                                               energies_in_kt=True)
            if np.isnan(pe):
                continue
            all_indices_to_keep.append(pos_index)
            all_energies.append(pe)

        rec_traj.save_hdf5(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_{sample_num}.h5")
        all_dihedrals = md.compute_dihedrals(rec_traj,
                                             full_z_matrix)
        np.save(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_dihedrals_{sample_num}.npy",
                all_dihedrals)
        np.save(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_energies_{sample_num}.npy", all_energies)
        np.save(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_log_prob_{sample_num}.npy",
                total_log_prob.detach().cpu().numpy())
        np.save(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_indices_to_keep_{sample_num}.npy", 
                all_indices_to_keep)
        np.save(f"{uncoupled_gmm_folder_name}/uncoupled_gmm_gen_components_{sample_num}.npy",
                all_gen_components)
