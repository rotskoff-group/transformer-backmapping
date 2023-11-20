import torch
import numpy as np
import argparse
import os
from sklearn.mixture import GaussianMixture
from cgp import GaussianMixtureModel

device = torch.device("cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--num_components", type=int, default=10)
parser.add_argument("--num_datapoints", type=int, default=50000)
parser.add_argument("--all_coordinates", action='store',
                    type=str, nargs="*", help='Residues to be included in the PDB file')
parser.add_argument("--protein_name", action='store',
                    type=str, help="Name of the protein")


config = parser.parse_args()
num_components = config.num_components
num_datapoints = config.num_datapoints
all_coordinates = config.all_coordinates
protein_name = config.protein_name


data_folder_name = f"./sidechain_info/{protein_name}/"

all_ic = []
all_offset = []
for coordinate in all_coordinates:
    coordinate_ic = np.load((data_folder_name + f"{coordinate}.npy"))
    if coordinate == "dihedrals":
        dihedrals_offset = np.load((data_folder_name + f"dihedrals_offset.npy"))
        coordinate_ic = coordinate_ic + dihedrals_offset
        to_subtract = (coordinate_ic > np.pi) * 2 * np.pi
        to_add = (coordinate_ic < -np.pi) * 2 * np.pi
        coordinate_ic -= to_subtract
        coordinate_ic += to_add
        all_offset.append(dihedrals_offset)
    else:
        all_offset.append(np.zeros((1, coordinate_ic.shape[-1])))
    all_ic.append(coordinate_ic)

all_ic = np.concatenate(all_ic, axis=-1)
all_offset = np.concatenate(all_offset, axis=-1)
in_dim = all_ic.shape[-1]


rand_indices = np.random.permutation(all_ic.shape[0])
all_ic = all_ic[rand_indices[0:num_datapoints]]


tag = (str(protein_name) + "_protein_name_"
       + "_".join(all_coordinates) + "_coordinate_"
       + str(num_components) + "_num_components_"
       + str(num_datapoints) + "_num_datapoints")


gm = GaussianMixture(n_components=num_components,
                     random_state=0, verbose=1, tol=1E-7, max_iter=1000,
                     verbose_interval=1).fit(all_ic)

sk_log_probs = gm.score_samples(all_ic)
sk_log_probs = sk_log_probs - sk_log_probs.max()
mean_sk_log_prob = sk_log_probs.mean()

sk_bic = gm.bic(all_ic)
sk_aic = gm.aic(all_ic)

sk_info = {"sk_log_probs": sk_log_probs,
           "sk_bic": sk_bic,
           "sk_aic": sk_aic}


means = torch.tensor(gm.means_, device=device).float()
covs = torch.tensor(gm.covariances_, device=device).float()
weights = torch.tensor(gm.weights_, device=device).float()

save_folder_name = "./" + tag + "/"
try:
    os.mkdir(save_folder_name)
except FileExistsError:
    pass


gmm_model = GaussianMixtureModel(means=means, covs=covs,
                                 weights=weights, 
                                 all_offset=all_offset,
                                 folder_name=save_folder_name,
                                 device=device)
gmm_model.save()


in_dim = all_ic.shape[-1]

gen_ic, _ = gmm_model.sample(1000)
gen_ic = gen_ic.cpu().numpy()

all_reverse_kl = []
all_forward_kl = []
for i in range(in_dim):
    gen_prob_dens, gen_bins = np.histogram(gen_ic[:, i], bins=np.arange(-np.pi, np.pi, 0.1), density=True)
    gen_probs = gen_prob_dens * np.diff(gen_bins)
    gen_probs += 1e-10
    data_prob_dens, data_bins = np.histogram(all_ic[:, i], bins=np.arange(-np.pi, np.pi, 0.1), density=True)
    data_probs = data_prob_dens * np.diff(data_bins)
    data_probs += 1e-10
    
    
    reverse_kl = np.sum(data_probs[data_probs > 0] * np.log(data_probs[data_probs > 0] / gen_probs[data_probs > 0]))
    forward_kl = np.sum(gen_probs[gen_probs > 0] * np.log(gen_probs[gen_probs > 0] / data_probs[gen_probs > 0]))
    
    all_reverse_kl.append(reverse_kl)
    all_forward_kl.append(forward_kl)
sk_info["reverse_kl"] = all_reverse_kl
sk_info["forward_kl"] = all_forward_kl
print(sk_info)
np.save(save_folder_name + "sk_info.npy", sk_info)