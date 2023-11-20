import numpy as np
import torch
import mdtraj as md
from . import reconstruct_atoms


class Factory:
    def __init__(self, lens):
        """
        Arguments:
            lens {Lens} -- Lens object containing information about the CG model
        """
        self.lens = lens

    def reconstruct_from_cg_traj(self, cg_traj_pos,
                                 ic_model,
                                 cg_c_info,
                                 z_matrix,
                                 grad_enabled=True,
                                 device=torch.device("cuda:0")):
        """Reconstruct all atoms from a CG trajectory
        Arguments:
            cg_traj_pos {torch.Tensor} -- CG trajectory positions
            dihedral_model {GenModel} -- Dihedral model
            bond_length_model {GenModel} -- Bond length model
            bond_angle_model {GenModel} -- Bond angle model
            cg_c_info {torch.Tensor} -- Conditional information for the CG trajectory
            z_matrix {np.ndarray} -- Z-matrix for the CG trajectory
        """
        all_reconstructed_positions_intermediate = cg_traj_pos
        all_fixed_atoms = self.lens.fixed_atoms_finer

        all_reconstructed_positions = torch.empty((all_reconstructed_positions_intermediate.shape[0],
                                                   all_reconstructed_positions_intermediate.shape[1] + z_matrix.shape[0], 3),
                                                  device=device)
        all_reconstructed_positions[:, :all_fixed_atoms.shape[0],
                                    :] = all_reconstructed_positions_intermediate

        with torch.set_grad_enabled(grad_enabled):
            gen_ic, total_log_prob = ic_model.sample(n_samples=cg_traj_pos.shape[0],
                                                     c_info=cg_c_info)

            gen_bond_lengths, gen_bond_angles, gen_dihedral_angles = torch.tensor_split(gen_ic,
                                                                                        [z_matrix.shape[0],
                                                                                         2*z_matrix.shape[0]], axis=-1)

        all_reconstructed_positions = reconstruct_atoms(all_reconstructed_positions=all_reconstructed_positions,
                                                        all_distances=gen_bond_lengths,
                                                        all_angles=gen_bond_angles,
                                                        all_dihedral_angles=gen_dihedral_angles,
                                                        fixed_atoms=all_fixed_atoms,
                                                        z_matrix=z_matrix,
                                                        device=device)

        return all_reconstructed_positions, total_log_prob

    def serial_reconstruct_from_cg_traj(self, cg_traj_pos,
                                        ic_models_by_res_num,
                                        cg_c_info_by_res_num,
                                        z_matrix_by_res_num,
                                        grad_enabled=True,
                                        return_gen_components=False,
                                        device=torch.device("cuda:0")):
        """Reconstruct each residue serially from a CG trajectory
        Arguments:
            cg_traj_pos {torch.Tensor} -- CG trajectory positions
            ic_models_by_res_num {dict} -- Dictionary mapping residue number to (length, angle, dihedral) GenModel
            cg_c_info_by_res_num {dict} -- Dictionary mapping residue number to conditional information
            z_matrix_by_res_num {dict} -- Dictionary mapping residue number to Z-matrix
            grad_enabled {bool} -- Whether to enable gradient calculation
            return_gen_components {bool} -- Whether to return the generated components
            device {torch.device} -- Device to use
        """
        all_reconstructed_positions_intermediate = cg_traj_pos
        all_fixed_atoms = self.lens.fixed_atoms_finer
        total_log_prob = 0
        all_gen_components = {}
        for res_num in z_matrix_by_res_num.keys():
            z_matrix = z_matrix_by_res_num[res_num]
            if z_matrix is None:
                continue
            cg_c_info = cg_c_info_by_res_num[res_num]

            ic_model = ic_models_by_res_num[res_num]

            all_reconstructed_positions = torch.empty((all_reconstructed_positions_intermediate.shape[0],
                                                       all_reconstructed_positions_intermediate.shape[1] + z_matrix.shape[0], 3), device=device)
            all_reconstructed_positions[:, :all_fixed_atoms.shape[0],
                                        :] = all_reconstructed_positions_intermediate

            with torch.set_grad_enabled(grad_enabled):
                gen_ic, total_log_prob_res = ic_model.sample(n_samples=cg_traj_pos.shape[0],
                                                         c_info=cg_c_info)
                
                if gen_ic.shape[-1] == ((3 * z_matrix.shape[0]) + 2):
                    gen_bond_lengths, gen_bond_angles, gen_dihedral_angles, gen_components = torch.tensor_split(gen_ic,
                                                                                                                [z_matrix.shape[0], 2*z_matrix.shape[0], 3*z_matrix.shape[0]], axis=-1)
                    gen_components = gen_components.detach().cpu().numpy()
                else:
                    gen_bond_lengths, gen_bond_angles, gen_dihedral_angles = torch.tensor_split(gen_ic,
                                                                                                [z_matrix.shape[0], 2*z_matrix.shape[0]], axis=-1)
                    gen_components = None
                all_gen_components[res_num] = gen_components


            all_reconstructed_positions = reconstruct_atoms(all_reconstructed_positions=all_reconstructed_positions,
                                                            all_distances=gen_bond_lengths,
                                                            all_angles=gen_bond_angles,
                                                            all_dihedral_angles=gen_dihedral_angles,
                                                            fixed_atoms=all_fixed_atoms,
                                                            z_matrix=z_matrix,
                                                            device=device)

            all_reconstructed_positions_intermediate = all_reconstructed_positions
            all_fixed_atoms = np.concatenate((all_fixed_atoms,
                                              z_matrix[:, 0]))
            all_fixed_atoms = np.sort(all_fixed_atoms)

            total_log_prob += total_log_prob_res

        all_reconstructed_positions = all_reconstructed_positions_intermediate
        if return_gen_components:
            return all_reconstructed_positions, total_log_prob, all_gen_components
        return all_reconstructed_positions, total_log_prob

