import numpy as np

class ResidueInfo:
    """A base class to store information about a residue's dihedral angles.
    """
    def __init__(self):
        """Initializes the ResidueInfo class.
        Attributes:
            phi: A list of atoms that make up the phi dihedral angle.
            psi: A list of atoms that make up the psi dihedral angle.
            omega: A list of atoms that make up the omega dihedral angle.
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        self.phi = ["C", "N", "CA", "C"]
        self.psi = ["N", "CA", "C", "N"]
        self.omega = ["CA", "C", "N", "CA"]
        self.dihedral_order = []
        self.all_offset = []
        self.n_components = 0

class ACEInfo(ResidueInfo):
    """A class to store information about the ACE residue's dihedral angles.
    """
    def __init__(self):
        """Initializes the ACEInfo class.
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [["O", "C", "CH3", "H1"],
                               ["O", "C", "CH3", "H2"],
                               ["O", "C", "CH3", "H3"]]
        self.all_offset = [[0, 0, 0]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 3


class NMEInfo(ResidueInfo):
    """A class to store information about the NME residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [["H", "N", "C", "H1"],
                               ["H", "N", "C", "H2"],
                               ["H", "N", "C", "H3"]]
        self.all_offset = [[0, 0, 0]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 3


class ALAInfo(ResidueInfo):
    """A class to store information about the ALA residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'HB1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3']]
        self.all_offset = [[np.pi/4, np.pi/4, np.pi/4]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 3

class ARGInfo(ResidueInfo):
    """A class to store information about the ARG residue's dihedral angles.
    """
    def __init__(self):
        super().__init__()
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD'],
                               ['CB', 'CG', 'CD', 'NE'],
                               ['CG', 'CD', 'NE', 'CZ'],
                               ['CD', 'NE', 'CZ', 'NH1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ['CB', 'CG', 'CD', 'HD2'],
                               ['CB', 'CG', 'CD', 'HD3'],
                               ['CG', 'CD', 'NE', 'HE'],
                               ['CD', 'NE', 'CZ', 'NH2'],
                               ['NE', 'CZ', 'NH1', 'HH11'],
                               ['NE', 'CZ', 'NH1', 'HH12'],
                               ['NE', 'CZ', 'NH2', 'HH21'],
                               ['NE', 'CZ', 'NH2', 'HH22']]
        self.all_offset = [[np.pi/4, np.pi, np.pi/4, np.pi, 0, np.pi/4, np.pi/4, np.pi/4, 
                            -np.pi/4, np.pi/3, np.pi/3, 0, np.pi/3, 0, np.pi/2, 0, np.pi/2]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 48

class ASNInfo(ResidueInfo):
    """A class to store information about the ASN residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'OD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'ND2'],
                               ['CB', 'CG', 'ND2', 'HD21'],
                               ['CB', 'CG', 'ND2', 'HD22']]
        self.all_offset = [[np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 16


class ASPInfo(ResidueInfo):
    """A class to store information about the ASP residue's dihedral angles.
    """
    def __init__(self):
        super().__init__()
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'OD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'OD2']]
        self.all_offset = [[np.pi/4, -np.pi/2, np.pi/4, np.pi/4, -np.pi/2]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 64



class CYSInfo(ResidueInfo):
    """A class to store information about the CYS residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'SG'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'SG', 'HG']]
        self.all_offset = [[np.pi/4, np.pi/4, np.pi/4, np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 8


class GLNInfo(ResidueInfo):
    """A class to store information about the GLN residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD'],
                               ['CB', 'CG', 'CD', 'OE1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ['CB', 'CG', 'CD', 'NE2'],
                               ['CG', 'CD', 'NE2', 'HE21'],
                               ['CG', 'CD', 'NE2', 'HE22']]
        self.all_offset = [[np.pi/4, np.pi/4, 0, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi, np.pi/2, 0]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 14


class GLUInfo(ResidueInfo):
    """A class to store information about the GLU residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD'],
                               ['CB', 'CG', 'CD', 'OE1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ['CB', 'CG', 'CD', 'OE2']]
        self.all_offset = [[-1, -1, -np.pi/2, -1, np.pi/3, np.pi/3, np.pi/3, -np.pi/2]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 64

class GLYInfo(ResidueInfo):
    """A class to store information about the GLY residue's dihedral angles.
    """
    def __init__(self):
        """Glycine does not have any dihedral angles.
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = []
        self.all_offset = []
        self.n_components = None


class HISInfo(ResidueInfo):
    """A class to store information about the HIS residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'ND1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD2'],
                               ['CB', 'CG', 'CD2', 'HD2'],
                               ['CB', 'CG', 'CD2', 'NE2'],
                               ['CB', 'CG', 'ND1', 'CE1'],
                               ['CG', 'CD2', 'NE2', 'HE2'],
                               ['CG', 'ND1', 'CE1', 'HE1']]
        self.all_offset = [[np.pi/4, -0.1, np.pi/3, np.pi/3, np.pi, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 8



class ILEInfo(ResidueInfo):
    """A class to store information about the ILE residue's dihedral angles.
    """
    def __init__(self):
        """ILE was not present in the proteins investigated so this class is incomplete.
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """        
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG1'],
                               ['CA', 'CB', 'CG1', 'CD1'],
                               ['N', 'CA', 'CB', 'CG2'],
                               ['N', 'CA', 'CB', 'HB'],
                               ['CA', 'CB', 'CG1', 'HG12'],
                               ['CA', 'CB', 'CG1', 'HG13'],
                               ['CA', 'CB', 'CG2', 'HG21'],
                               ['CA', 'CB', 'CG2', 'HG22'],
                               ['CA', 'CB', 'CG2', 'HG23'],
                               ['CB', 'CG1', 'CD1', 'HD11'],
                               ['CB', 'CG1', 'CD1', 'HD12'],
                               ['CB', 'CG1', 'CD1', 'HD13']]
        # self.all_offset = []
        # assert len(self.dihedral_order) == len(self.all_offset[0])
        # self.n_components = 0

class LEUInfo(ResidueInfo):
    """A class to store information about the LEU residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """        
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD2'],
                               ['CA', 'CB', 'CG', 'HG'],
                               ['CB', 'CG', 'CD1', 'HD11'],
                               ['CB', 'CG', 'CD1', 'HD12'],
                               ['CB', 'CG', 'CD1', 'HD13'],
                               ['CB', 'CG', 'CD2', 'HD21'],
                               ['CB', 'CG', 'CD2', 'HD22'],
                               ['CB', 'CG', 'CD2', 'HD23']]
        
        self.all_offset = [[1, -np.pi/3, -1, -np.pi/3, np.pi/3, 0.5, np.pi/4, np.pi/4, np.pi/4, np.pi/3, np.pi/4, np.pi/4]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 64

class LYSInfo(ResidueInfo):
    """A class to store information about the LYS residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD'],
                               ['CB', 'CG', 'CD', 'CE'],
                               ['CG', 'CD', 'CE', 'NZ'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ['CB', 'CG', 'CD', 'HD2'],
                               ['CB', 'CG', 'CD', 'HD3'],
                               ['CG', 'CD', 'CE', 'HE2'],
                               ['CG', 'CD', 'CE', 'HE3'],
                               ['CD', 'CE', 'NZ', 'HZ1'],
                               ['CD', 'CE', 'NZ', 'HZ2'],
                               ['CD', 'CE', 'NZ', 'HZ3']]

        self.all_offset = [[np.pi, np.pi, np.pi, np.pi, -1, -1, 1, -1, 1, -1, 1, -1, np.pi, np.pi, np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 96
class METInfo(ResidueInfo):
    """A class to store information about the MET residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'SD'],
                               ["CB", "CG", "SD", "CE"],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ["CG", "SD", "CE", "HE1"],
                               ["CG", "SD", "CE", "HE2"],
                               ["CG", "SD", "CE", "HE3"]]
        self.all_offset = [[np.pi, np.pi, np.pi, -1, -1, 1, -1, np.pi, np.pi, np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 64

class PHEInfo(ResidueInfo):
    """A class to store information about the PHE residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD2'],
                               ['CB', 'CG', 'CD1', 'CE1'],
                               ['CB', 'CG', 'CD1', 'HD1'],
                               ['CB', 'CG', 'CD2', 'HD2'],
                               ['CG', 'CD1', 'CE1', 'CZ'],
                               ['CG', 'CD1', 'CE1', 'HE1'],
                               ['CB', 'CG', 'CD2', 'CE2'],
                               ['CD1', 'CE1', 'CZ', 'HZ'],
                               ['CG', 'CD2', 'CE2', 'HE2']]
        self.all_offset = [[np.pi/4, 0, np.pi/4, np.pi/4, 0, np.pi/4, 0, 0, 0, np.pi/4, np.pi/4, np.pi/4, np.pi/4]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 8

class PROInfo(ResidueInfo):
    """A class to store information about the PRO residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD'],
                               ['CA', 'CB', 'CG', 'HG2'],
                               ['CA', 'CB', 'CG', 'HG3'],
                               ['CB', 'CG', 'CD', 'HD2'],
                               ['CB', 'CG', 'CD', 'HD3']]
        self.all_offset = [[0, 0, 0, 0, 0, 0, 0, 0]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 2


class SERInfo(ResidueInfo):
    """A class to store information about the SER residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'OG'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'OG', 'HG']]
        self.all_offset = [[np.pi/4, np.pi/4, np.pi/4, np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 12


class THRInfo(ResidueInfo):
    """A class to store information about the THR residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'OG1'],
                               ['N', 'CA', 'CB', 'CG2'],
                               ['N', 'CA', 'CB', 'HB'],
                               ['CA', 'CB', 'CG2', 'HG21'],
                               ['CA', 'CB', 'CG2', 'HG22'],
                               ['CA', 'CB', 'CG2', 'HG23'],
                               ['CA', 'CB', 'OG1', 'HG1']]
        self.all_offset = [[0.5, np.pi/4, np.pi/3, np.pi/3, np.pi/3, np.pi/3, -np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 32



class TRPInfo(ResidueInfo):
    """A class to store information about the TRP residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD2'],
                               ['CB', 'CG', 'CD1', 'HD1'],
                               ['CB', 'CG', 'CD1', 'NE1'],
                               ['CB', 'CG', 'CD2', 'CE3'],
                               ['CB', 'CG', 'CD2', 'CE2'],
                               ['CG', 'CD1', 'NE1', 'HE1'],
                               ['CG', 'CD2', 'CE3', 'CZ3'],
                               ['CG', 'CD2', 'CE3', 'HE3'],
                               ['CG', 'CD2', 'CE2', 'CZ2'],
                               ['CD2', 'CE3', 'CZ3', 'HZ3'],
                               ['CD2', 'CE2', 'CZ2', 'CH2'],
                               ['CD2', 'CE2', 'CZ2', 'HZ2'],
                               ['CE2', 'CZ2', 'CH2', 'HH2']]
        
        self.all_offset = [[np.pi/3, -np.pi/6, -1, -1, 2.9, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 8

class TYRInfo(ResidueInfo):
    """A class to store information about the TYR residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG'],
                               ['CA', 'CB', 'CG', 'CD1'],
                               ['N', 'CA', 'CB', 'HB2'],
                               ['N', 'CA', 'CB', 'HB3'],
                               ['CA', 'CB', 'CG', 'CD2'],
                               ['CB', 'CG', 'CD1', 'CE1'],
                               ['CB', 'CG', 'CD1', 'HD1'],
                               ['CB', 'CG', 'CD2', 'HD2'],
                               ['CG', 'CD1', 'CE1', 'CZ'],
                               ['CG', 'CD1', 'CE1', 'HE1'],
                               ['CB', 'CG', 'CD2', 'CE2'],
                               ['CD1', 'CE1', 'CZ', 'OH'],
                               ['CG', 'CD2', 'CE2', 'HE2'],
                               ['CE1', 'CZ', 'OH', 'HH']]
        self.all_offset = [[-1, 0, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, -1, -np.pi/2]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 16


class VALInfo(ResidueInfo):
    """A class to store information about the VAL residue's dihedral angles.
    """
    def __init__(self):
        """
        Attributes:
            dihedral_order: A list of lists of atoms that make up the dihedral angles.
            all_offset: A list of lists of floats that represent the offset of each dihedral angle.
            n_components: An integer representing the number of components in the dihedral angles.
        """   
        super().__init__()
        self.dihedral_order = [['N', 'CA', 'CB', 'CG1'],
                               ['N', 'CA', 'CB', 'CG2'],
                               ['N', 'CA', 'CB', 'HB'],
                               ['CA', 'CB', 'CG1', 'HG11'],
                               ['CA', 'CB', 'CG1', 'HG12'],
                               ['CA', 'CB', 'CG1', 'HG13'],
                               ['CA', 'CB', 'CG2', 'HG21'],
                               ['CA', 'CB', 'CG2', 'HG22'],
                               ['CA', 'CB', 'CG2', 'HG23']]
        self.all_offset = [[np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]]
        assert len(self.dihedral_order) == len(self.all_offset[0])
        self.n_components = 32


res_code_to_res_info = {"ACE": ACEInfo(),
                        "NME": NMEInfo(),
                        "ALA": ALAInfo(),
                        "ARG": ARGInfo(),
                        "ASN": ASNInfo(),
                        "ASP": ASPInfo(),
                        "CYS": CYSInfo(),
                        "GLN": GLNInfo(),
                        "GLU": GLUInfo(),
                        "GLY": GLYInfo(),
                        "HIS": HISInfo(),
                        "ILE": ILEInfo(),
                        "LEU": LEUInfo(),
                        "LYS": LYSInfo(),
                        "MET": METInfo(),
                        "PHE": PHEInfo(),
                        "PRO": PROInfo(),
                        "SER": SERInfo(),
                        "THR": THRInfo(),
                        "TRP": TRPInfo(),
                        "TYR": TYRInfo(),
                        "VAL": VALInfo()}
