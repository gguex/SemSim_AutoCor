from code_python.local_functions import sim_to_dissim, exchange_and_transition_matrices, \
    discontinuity_segmentation, write_groups_in_html_file, write_membership_mat_in_csv_file
import numpy as np
import os
from sklearn.metrics import normalized_mutual_info_score

# --- Parameters

input_file = "mix_sent1_min5.txt"
sim_tag = "wesim"
dist_option = "minus_log"
exch_mat_opt = "s"
exch_range = 3
n_groups = 4
alpha = 5
beta = 50
kappa = 2/3

# --- Paths

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
# Defining the real group file
real_group_file = f"{base_path}/corpora/mixgroup_{input_file[4:]}"

# --- Computations

# Compute the dissimilartiy_matrix
d_ext_mat, token_list = sim_to_dissim(input_file=input_file,
                                      sim_tag=sim_tag,
                                      dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                   exch_mat_opt=exch_mat_opt,
                                                   exch_range=exch_range)

# Compute the membership matrix
result_matrix = discontinuity_segmentation(d_ext_mat=d_ext_mat,
                                           exch_mat=exch_mat,
                                           w_mat=w_mat,
                                           n_groups=n_groups,
                                           alpha=alpha,
                                           beta=beta,
                                           kappa=kappa)
# Write html results
write_groups_in_html_file("test.html", token_list, result_matrix)
# Write csv results
write_membership_mat_in_csv_file("test.csv", token_list, result_matrix)

# Real results
with open(real_group_file) as group_file:
    real_group_vec = group_file.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
z_real_mat = np.zeros((len(token_list), n_groups))
for i, label in enumerate(real_group_vec):
    if label != 0:
        z_real_mat[i, :] = 0
        z_real_mat[i, label - 1] = 1

# Write html results
write_groups_in_html_file("test_real.html", token_list, z_real_mat)

# Compute the groups
algo_group_value = np.argmax(result_matrix, 1) + 1

# Compute nmi score
nmi = normalized_mutual_info_score(real_group_vec, algo_group_value)
print(nmi)
