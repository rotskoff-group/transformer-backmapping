from cgp import ProteinSolvent
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--protein_name", action='store',
                    type=str, help="Name of the protein")
parser.add_argument("--num_data_points", type=int, default=100)
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--friction", type=float, default=0.1)
parser.add_argument("--save_freq", type=int, default=10000)
parser.add_argument("--use_hbond_constraints", action='store_true', default=False)


config = parser.parse_args()
protein_name = config.protein_name
num_data_points = config.num_data_points
save_freq = config.save_freq

dt = config.dt
friction = config.friction
use_hbond_constraints = config.use_hbond_constraints


# chk = config.chk


#get all files ending with .chk
chk_files = [f for f in os.listdir() if f.endswith(".chk")]
chk_files = [f for f in chk_files if f[0:len(protein_name)] == protein_name]
if len(chk_files) == 0:
    max_chk = 0
else:
    max_chk = max([int(f.split("_")[-1].split(".")[0]) for f in chk_files]) + 1


omm = ProteinSolvent(filename=protein_name, chk=max_chk, save_int=10000, use_hbond_constraints=use_hbond_constraints, 
                     dt=dt, friction=friction)
omm.generate_long_trajectory(num_data_points=num_data_points, save_freq=save_freq)

