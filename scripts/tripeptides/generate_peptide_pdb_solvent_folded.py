import argparse
import os
from openmm import app



parser = argparse.ArgumentParser()
parser.add_argument('--residues', action='store', type=str, nargs="*", help='Residues to be included in the PDB file')
parser.add_argument("--protein_name", action='store', type=str, help="Name of the protein")

config = parser.parse_args()
residues = config.residues
protein_name = config.protein_name

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
    f.write("source oldff/leaprc.ff14SB\n")
    f.write("source leaprc.water.tip3p\n")
    f.write("set default PBRadii mbondi2\n")
    f.write("peptide = sequence { " + " ".join(residues) + " }\n")
    f.write("check peptide\n")
    f.write("charge peptide\n")
    f.write("solvateBox peptide TIP3PBOX 9.0 iso\n")
    f.write(f"saveAmberParm peptide {filename[:-3]}.prmtop {filename[:-3]}.crd\n")
    f.write("quit\n")


os.system("tleap -f " + filename)

#Convert amber prmtop and crd to pdb
prmtop = app.AmberPrmtopFile(filename[:-3] + ".prmtop")
inpcrd = app.AmberInpcrdFile(filename[:-3] + ".crd")

#Write to pdb
with open(filename[:-3] + ".pdb", "w") as f:
    app.PDBFile.writeFile(prmtop.topology, inpcrd.getPositions(), f, keepIds=False)