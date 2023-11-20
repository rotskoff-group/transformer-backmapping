import argparse
import os
from openmm import app
import parmed as pmd



parser = argparse.ArgumentParser()
parser.add_argument('--residues', action='store', type=str, nargs="*", help='Residues to be included in the PDB file')
parser.add_argument('--residues_file', type=argparse.FileType('r'))
parser.add_argument("--protein_name", action='store', type=str, help="Name of the protein")

config = parser.parse_args()
residues = config.residues
protein_name = config.protein_name
residues_file = config.residues_file

if residues_file is not None:
    assert residues is None
    with residues_file as file:
        residues = [line.rstrip() for line in file]


else:
    assert residues is not None

if protein_name is not None:
    filename = protein_name + ".in"
else:
    #Create a string of residues    
    filename = "_".join(residues)
    filename = filename + ".in"

residues = ["ACE"] + residues
residues = residues + ["NME"]

#Write the string to filename
with open(filename, "w") as f:
    f.write("source leaprc.protein.ff99SBdisp\n")
    f.write("source leaprc.water.tip4pd_disp\n")
    f.write("set default PBRadii mbondi2\n")
    f.write("peptide = sequence { " + " ".join(residues) + " }\n")
    f.write("check peptide\n")
    f.write("charge peptide\n")
    f.write("solvateBox peptide TIP4PEWBOX 9.0 iso\n")
    f.write(f"saveAmberParm peptide {filename[:-3]}.prmtop {filename[:-3]}.crd\n")
    f.write("quit\n")


os.system("tleap -f " + filename)

#Convert amber prmtop and crd to pdb
prmtop = app.AmberPrmtopFile(filename[:-3] + ".prmtop")
inpcrd = app.AmberInpcrdFile(filename[:-3] + ".crd")


parm = pmd.amber.AmberFormat(filename[:-3] + ".prmtop")

for i, (per, k) in enumerate(zip(parm.parm_data["DIHEDRAL_PERIODICITY"], parm.parm_data["DIHEDRAL_FORCE_CONSTANT"])):
    if per == 0:
        parm.parm_data["DIHEDRAL_PERIODICITY"][i] = 1  # value doesn't matter
        parm.parm_data["DIHEDRAL_FORCE_CONSTANT"][i] = 0.0  # value doesn't matter
parm.write_parm(filename[:-3] + ".prmtop")


#Write to pdb
with open(filename[:-3] + ".pdb", "w") as f:
    app.PDBFile.writeFile(prmtop.topology, inpcrd.getPositions(), f, keepIds=False)