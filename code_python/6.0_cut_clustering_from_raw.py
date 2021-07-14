from local_functions import *
from os.path import expanduser
import pandas as pd

input_file = "Sidelights_on_relativity.txt"

sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 10
alpha = 5
beta = 50
kappa = 0.8

block_size = 1000
n_groups = 5

# Getting the file path
working_path = os.getcwd()
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
file_path = f"{base_path}/corpora/{input_file}"

# Getting the wv path
home = expanduser("~")
wv_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"

z_res, existing_token_list = cut_clustering_from_raw(file_path, wv_path, dist_option, exch_mat_opt, exch_range, n_groups, alpha, beta,
                              kappa, block_size=block_size, verbose=True)

write_groups_in_html_file("test.html", existing_token_list, z_res)

# Compute the aggregate labels
df_results = pd.DataFrame(z_res)
df_results["Token"] = existing_token_list
type_results = df_results.groupby("Token").mean()
type_list = list(type_results.index)
type_values = type_results.to_numpy()

write_membership_mat_in_csv_file("test.csv", type_list, type_values)