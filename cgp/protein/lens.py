import numpy as np
import mdtraj as md
from . import res_code_to_res_info

"""Classes to investigate properties of proteins at different "lenses"
"""


class Lens:
    def __init__(self, protein_top,
                 topology_atom_names_finer,
                 topology_atom_names_coarser):
        """
        Arguments:
            protein_top {mdtraj.Topology} -- topology of full protein (all atoms)
            topology_atom_names_finer {list} -- list of atom names to keep
            topology_atom_names_coarser {list} -- list of atom names to keep
        """

        assert protein_top.residue(0).name == "ACE"
        assert (protein_top.residue(-1).name == "NME"
                or protein_top.residue(-1).name == "NH2")

        self.atom_indices_finer = np.array([atom.index for atom in protein_top.atoms
                                            if atom.name in topology_atom_names_finer])
        self.topology_finer = protein_top.subset(self.atom_indices_finer)

        self.fixed_atoms_finer = np.array([atom.index for atom in self.topology_finer.atoms
                                           if atom.name in topology_atom_names_coarser])
        self.atom_names_finer = np.array([[atom.name, atom.residue.index, atom.index]
                                          for atom in self.topology_finer.atoms])

        self.atom_indices_coarser = [atom.index for atom in protein_top.atoms
                                     if atom.name in topology_atom_names_coarser]
        self.topology_coarser = protein_top.subset(self.atom_indices_coarser)
        self.atom_names_coarser = np.array([[atom.name, atom.residue.index, atom.index]
                                            for atom in self.topology_coarser.atoms])

    def _compute_data_from_traj(self, traj, dihedral_indices):
        """Computes dihedrals, distances, and angles from a trajectory
        Arguments:
            traj {mdtraj.Trajectory} -- trajectory to compute data from
            dihedral_indices {np.array} -- indices of atoms to compute dihedrals from
        """
        dihedrals = md.compute_dihedrals(traj, dihedral_indices)
        distances = md.compute_distances(traj,
                                         dihedral_indices[:, 0:2])
        angles = md.compute_angles(traj, dihedral_indices[:, 0:3])
        return dihedrals, distances, angles

    def _get_labeled_backbone_atom_indices(self):
        raise NotImplementedError

    def get_data(self, protein_traj):
        raise NotImplementedError


class BackboneLens(Lens):
    """Lens for going between core backbone ([N, CA, C, CH3]) 
       and full backbone ([N, CA, C, CH3, O, CB, HA, HA2, HA3]
    """
    def __init__(self, protein_top):
        """
        Arguments:
            protein_top {mdtraj.Topology} -- topology of full protein (all atoms)
        """
        topology_atom_names_finer = ["N", "H", "CA", "C", "O",
                                     "CB", "HA", "CH3", "HA2", "HA3"]
        topology_atom_names_coarser = ["N", "CA", "C", "CH3"]
        super().__init__(protein_top=protein_top,
                         topology_atom_names_finer=topology_atom_names_finer,
                         topology_atom_names_coarser=topology_atom_names_coarser)

    def _get_labeled_backbone_atom_indices(self, max_residue):
        """Gets labeled backbone atom indices for a protein
        Arguments:
            max_residue {int} -- maximum residue number to consider
        Returns:
            dict -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
        """
        N_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "N"]
        H_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "H"]
        C_atoms = self.atom_names_finer[((self.atom_names_finer[:, 0] == "C")
                                         & (self.atom_names_finer[:, 1].astype(int) < max_residue))]
        O_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "O"]
        CA_atoms = self.atom_names_finer[(self.atom_names_finer[:, 0] == "CA")
                                         | ((self.atom_names_finer[:, 0] == "C") & (self.atom_names_finer[:, 1].astype(int) == max_residue))
                                         | ((self.atom_names_finer[:, 0] == "CH3") & (self.atom_names_finer[:, 1].astype(int) == 0))]
        HA_atoms = self.atom_names_finer[(self.atom_names_finer[:, 0] == "HA")
                                         | (self.atom_names_finer[:, 0] == "HA2")]
        CB_atoms = self.atom_names_finer[(self.atom_names_finer[:, 0] == "CB")
                                         | (self.atom_names_finer[:, 0] == "HA3")]
        labeled_backbone_atom_indices = {"N": N_atoms,
                                         "H": H_atoms,
                                         "C": C_atoms,
                                         "O": O_atoms,
                                         "CA": CA_atoms,
                                         "HA": HA_atoms,
                                         "CB": CB_atoms}
        return labeled_backbone_atom_indices

    def _get_n_h_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for N-H dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
        Returns:
            np.array -- dihedral indices for N-H dihedrals
        """
        all_H_atoms = labeled_backbone_atom_indices["H"]
        all_N_atoms = labeled_backbone_atom_indices["N"]
        all_C_atoms = labeled_backbone_atom_indices["C"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]

        all_residues_to_consider = all_H_atoms[:, 1].astype("int")
        first_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num - 1)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = all_H_atoms[:, -1].astype("int")
        dihedral_indices = np.stack([first_indices, second_indices,
                                     third_indices, fourth_indices], axis=-1).astype("int")
        dihedral_indices = np.flip(dihedral_indices, axis=-1)
        return dihedral_indices

    def _get_c_o_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for C-O dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
        Returns:
            np.array -- dihedral indices for C-O dihedrals
        """
        all_O_atoms = labeled_backbone_atom_indices["O"]
        all_C_atoms = labeled_backbone_atom_indices["C"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]
        all_N_atoms = labeled_backbone_atom_indices["N"]

        all_residues_to_consider = all_O_atoms[:, 1].astype("int")
        first_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num + 1)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = np.array([all_O_atoms[all_O_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        dihedral_indices = np.stack([first_indices, second_indices,
                                     third_indices, fourth_indices], axis=-1).astype("int")
        dihedral_indices = np.flip(dihedral_indices, axis=-1)
        return dihedral_indices

    def _get_ca_cb_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for CA-CB dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
        Returns:
            np.array -- dihedral indices for CA-CB dihedrals
        """
        all_CB_atoms = labeled_backbone_atom_indices["CB"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]
        all_N_atoms = labeled_backbone_atom_indices["N"]
        all_C_atoms = labeled_backbone_atom_indices["C"]

        all_residues_to_consider = all_CB_atoms[:, 1].astype("int")
        first_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = np.array([all_CB_atoms[all_CB_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        dihedral_indices = np.stack([first_indices, second_indices,
                                     third_indices, fourth_indices], axis=-1).astype("int")
        dihedral_indices = np.flip(dihedral_indices, axis=-1)
        return dihedral_indices

    def _get_ca_ha_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for CA-HA dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
        Returns:
            np.array -- dihedral indices for CA-HA dihedrals
        """
        all_HA_atoms = labeled_backbone_atom_indices["HA"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]
        all_N_atoms = labeled_backbone_atom_indices["N"]
        all_C_atoms = labeled_backbone_atom_indices["C"]

        all_residues_to_consider = all_HA_atoms[:, 1].astype("int")
        first_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = np.array([all_HA_atoms[all_HA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        dihedral_indices = np.stack([first_indices, second_indices,
                                     third_indices, fourth_indices], axis=-1).astype("int")
        dihedral_indices = np.flip(dihedral_indices, axis=-1)
        return dihedral_indices

    def _get_x_x_info(self, get_x_x_dihedrals,
                      labeled_backbone_atom_indices,
                      full_backbone_traj):
        """Gets dihedral, distance, and angle information for a given set of dihedrals
        Arguments:
            get_x_x_dihedrals {function} -- function to get dihedral indices
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, H, C, O, CA, HA, CB)
            full_backbone_traj {mdtraj.Trajectory} -- trajectory to compute data from
        Returns:
            dict -- dictionary of dihedral, distance, angle, and dihedral indices
        """
        dihedral_indices = get_x_x_dihedrals(
            labeled_backbone_atom_indices=labeled_backbone_atom_indices)
        dihedrals, distances, angles = self._compute_data_from_traj(full_backbone_traj,
                                                                    dihedral_indices)
        x_x_info = {"dihedrals": dihedrals,
                    "distances": distances,
                    "angles": angles,
                    "dihedral_indices": dihedral_indices}
        return x_x_info

    def get_data(self, protein_traj):
        """Gets dihedral, distance, and angle information for backbone atoms from atomistic trajectory
        Arguments:
            protein_traj {mdtraj.Trajectory} -- atomistic trajectory to compute data from
        Returns:
            dict -- dictionary of dihedral, distance, angle, and dihedral indices for n_h, c_o, ca_cb, and ca_ha
        """
        full_backbone_traj = protein_traj.atom_slice(self.atom_indices_finer)
        assert full_backbone_traj.topology == self.topology_finer
        max_residue = self.topology_finer.n_residues - 1
        labeled_backbone_atom_indices = self._get_labeled_backbone_atom_indices(
            max_residue)

        n_h_info = self._get_x_x_info(get_x_x_dihedrals=self._get_n_h_dihedrals,
                                      labeled_backbone_atom_indices=labeled_backbone_atom_indices,
                                      full_backbone_traj=full_backbone_traj)
        c_o_info = self._get_x_x_info(get_x_x_dihedrals=self._get_c_o_dihedrals,
                                      labeled_backbone_atom_indices=labeled_backbone_atom_indices,
                                      full_backbone_traj=full_backbone_traj)
        ca_cb_info = self._get_x_x_info(get_x_x_dihedrals=self._get_ca_cb_dihedrals,
                                        labeled_backbone_atom_indices=labeled_backbone_atom_indices,
                                        full_backbone_traj=full_backbone_traj)
        ca_ha_info = self._get_x_x_info(get_x_x_dihedrals=self._get_ca_ha_dihedrals,
                                        labeled_backbone_atom_indices=labeled_backbone_atom_indices,
                                        full_backbone_traj=full_backbone_traj)
        backbone_info = {"n_h": n_h_info,
                         "c_o": c_o_info,
                         "ca_cb": ca_cb_info,
                         "ca_ha": ca_ha_info}
        return backbone_info


class SideChainLens(Lens):
    """Lens for going between full backbone ([N, CA, C, CH3, O, CB, HA, HA2, HA3]
    and full protein
    """
    def __init__(self, protein_top):
        """
        Arguments:
            protein_top {mdtraj.Topology} -- topology of full protein (all atoms)
        """
        topology_atom_names_coarser = ["N", "H", "CA", "C", "O",
                                       "CB", "HA", "CH3", "HA2", "HA3"]
        topology_atom_names_finer = [atom.name for atom in protein_top.atoms]

        super().__init__(protein_top=protein_top,
                         topology_atom_names_finer=topology_atom_names_finer,
                         topology_atom_names_coarser=topology_atom_names_coarser)

    def _get_labeled_backbone_atom_indices(self):
        """Gets labeled backbone atom indices for a protein (atomistic)
        Returns:
            dict -- dictionary of atom names for each backbone atom (N, C, CA)
        """
        N_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "N"]
        C_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "C"]
        CA_atoms = self.atom_names_finer[self.atom_names_finer[:, 0] == "CA"]
        labeled_backbone_atom_indices = {"N": N_atoms,
                                         "C": C_atoms,
                                         "CA": CA_atoms}
        return labeled_backbone_atom_indices

    def _get_labeled_backbone_atom_indices_coarser(self):
        """Gets labeled backbone atom indices for a protein (full backbone only)
        Returns:
            dict -- dictionary of atom names for each backbone atom (N, C, CA)
        """
        N_atoms = self.atom_names_coarser[self.atom_names_coarser[:, 0] == "N"]
        C_atoms = self.atom_names_coarser[self.atom_names_coarser[:, 0] == "C"]
        CA_atoms = self.atom_names_coarser[self.atom_names_coarser[:, 0] == "CA"]
        labeled_backbone_atom_indices = {"N": N_atoms,
                                         "C": C_atoms,
                                         "CA": CA_atoms}
        return labeled_backbone_atom_indices

    def _get_chi_dihedrals(self, dihedral_order, atom_names_finer_residue):
        """Gets dihedral indices for chi dihedrals
        Arguments:
            dihedral_order {list} -- list of lists of atom names for each chi dihedral
            atom_names_finer_residue {np.array} -- array of atom names for each atom in a residue
        Returns:
            np.array -- dihedral indices for chi dihedrals
        """
        assert np.all(atom_names_finer_residue[0, -2]
                      == atom_names_finer_residue[:, -2])
        all_chi_dihedral_indices = []
        for dihedral_atom_names in dihedral_order:
            chi_dihedral_indices = []
            for dihedral_atom_name in dihedral_atom_names:
                atom_info = atom_names_finer_residue[atom_names_finer_residue[:, 0]
                                                     == dihedral_atom_name]
                assert len(atom_info) == 1
                chi_dihedral_indices.append(atom_info[0][-1])
            all_chi_dihedral_indices.append(chi_dihedral_indices)
        all_chi_dihedral_indices = np.array(
            all_chi_dihedral_indices).astype("int")
        all_chi_dihedral_indices = np.flip(all_chi_dihedral_indices, axis=-1)
        return all_chi_dihedral_indices

    def _get_phi_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for phi dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, C, CA)
        Returns:
            np.array -- dihedral indices for phi dihedrals
        """
        all_C_atoms = labeled_backbone_atom_indices["C"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]
        all_N_atoms = labeled_backbone_atom_indices["N"]

        all_residues_to_consider = all_CA_atoms[:, 1].astype("int")
        first_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num - 1)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        phi_dihedral_indices = np.stack([first_indices, second_indices,
                                         third_indices, fourth_indices], axis=-1).astype("int")
        return phi_dihedral_indices

    def _get_psi_dihedrals(self, labeled_backbone_atom_indices):
        """Gets dihedral indices for psi dihedrals
        Arguments:
            labeled_backbone_atom_indices {dict} -- dictionary of atom names for each backbone atom (N, C, CA)
        Returns:
            np.array -- dihedral indices for psi dihedrals
        """
        all_C_atoms = labeled_backbone_atom_indices["C"]
        all_CA_atoms = labeled_backbone_atom_indices["CA"]
        all_N_atoms = labeled_backbone_atom_indices["N"]

        all_residues_to_consider = all_CA_atoms[:, 1].astype("int")
        first_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        second_indices = np.array([all_CA_atoms[all_CA_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                   for residue_num in all_residues_to_consider])
        third_indices = np.array([all_C_atoms[all_C_atoms[:, -2].astype('int') == (residue_num)][0][-1]
                                  for residue_num in all_residues_to_consider])
        fourth_indices = np.array([all_N_atoms[all_N_atoms[:, -2].astype('int') == (residue_num + 1)][0][-1]
                                   for residue_num in all_residues_to_consider])
        psi_dihedral_indices = np.stack([first_indices, second_indices,
                                         third_indices, fourth_indices], axis=-1).astype("int")
        return psi_dihedral_indices

    def _get_info(self, protein_traj, dihedral_indices):
        """Gets dihedral, distance, and angle information for a given set of dihedrals
        Arguments:
            protein_traj {mdtraj.Trajectory} -- trajectory to compute data from
            dihedral_indices {np.array} -- indices of atoms to compute dihedrals from
        Returns:
            dict -- dictionary of dihedral, distance, angle, and dihedral indices
        """
        dihedrals, distances, angles = self._compute_data_from_traj(traj=protein_traj,
                                                                    dihedral_indices=dihedral_indices)
        info = {"dihedrals": dihedrals,
                "distances": distances,
                "angles": angles,
                "dihedral_indices": dihedral_indices}
        return info

    def get_data(self, protein_traj):
        """Gets dihedral, distance, and angle information for side chain atoms from atomistic trajectory
        Arguments:
            protein_traj {mdtraj.Trajectory} -- atomistic trajectory to compute data from   
        Returns:   
            dict -- dictionary of dihedral, distance, angle, and dihedral indices for psi, phi, and chi
        """
        assert protein_traj.topology == self.topology_finer
        labeled_backbone_atom_indices = self._get_labeled_backbone_atom_indices()

        psi_dihedral_indices = self._get_psi_dihedrals(
            labeled_backbone_atom_indices=labeled_backbone_atom_indices)
        phi_dihedral_indices = self._get_phi_dihedrals(
            labeled_backbone_atom_indices=labeled_backbone_atom_indices)

        psi_info = self._get_info(protein_traj=protein_traj,
                                  dihedral_indices=psi_dihedral_indices)
        phi_info = self._get_info(protein_traj=protein_traj,
                                  dihedral_indices=phi_dihedral_indices)

        all_chi_info_by_residue_num = {}
        for residue_num in range(0, protein_traj.n_residues):
            atom_names_finer_residue = self.atom_names_finer[self.atom_names_finer[:, 1].astype(int)
                                                             == residue_num]

            dihedral_order = res_code_to_res_info[protein_traj.top.residue(
                residue_num).name].dihedral_order
            all_chi_dihedral_indices = self._get_chi_dihedrals(dihedral_order=dihedral_order,
                                                               atom_names_finer_residue=atom_names_finer_residue)
            if all_chi_dihedral_indices.shape[0] == 0:
                all_chi_info_by_residue_num[residue_num] = None
            else:
                chi_info = self._get_info(protein_traj=protein_traj,
                                          dihedral_indices=all_chi_dihedral_indices)
                all_chi_info_by_residue_num[residue_num] = chi_info

        side_chain_info = {"psi": psi_info,
                           "phi": phi_info,
                           "chi": all_chi_info_by_residue_num}
        return side_chain_info

    def get_data_cg(self, cg_traj):
        """Gets dihedral, distance, and angle information of backbone from coarse-grained trajectory
        Arguments:
            cg_traj {mdtraj.Trajectory} -- coarse-grained trajectory to compute data from
        Returns:
            dict -- dictionary of dihedral, distance, angle, and dihedral indices for psi, ph
        """
        assert cg_traj.topology == self.topology_coarser
        labeled_backbone_atom_indices = self._get_labeled_backbone_atom_indices_coarser()

        psi_dihedral_indices = self._get_psi_dihedrals(
            labeled_backbone_atom_indices=labeled_backbone_atom_indices)
        phi_dihedral_indices = self._get_phi_dihedrals(
            labeled_backbone_atom_indices=labeled_backbone_atom_indices)

        psi_info = self._get_info(protein_traj=cg_traj,
                                  dihedral_indices=psi_dihedral_indices)
        phi_info = self._get_info(protein_traj=cg_traj,
                                  dihedral_indices=phi_dihedral_indices)

        c_info = {"psi": psi_info,
                  "phi": phi_info}
        return c_info
