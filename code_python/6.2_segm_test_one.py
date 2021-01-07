from code_python.local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, discontinuity_segmentation, cut_segmentation, \
    write_groups_in_html_file, write_membership_mat_in_csv_file
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

# -------------------------------------
# --- Parameters
# -------------------------------------

input_file = "mix_word1.txt"

output_html = "best_word1_semi10.html"
real_html = "word1_real.html"
token_csv = "best_word1_semi10.csv"
type_csv = "best_word1_semi10_type.csv"

sim_tag = "wesim"
dist_option = "minus_log"
exch_mat_opt = "d"
exch_range = 15
n_groups = 4
alpha = 0.1
beta = 300
kappa = 1
known_label_ratio = 0.1 # if 0, clustering
# segm_function = discontinuity_segmentation
segm_function = cut_segmentation
max_it = 1000

# -------------------------------------
# --- Loading and preprocessing
# -------------------------------------

# Get the file paths
text_file_path, typefreq_file_path, sim_file_path, ground_truth_path = get_all_paths(input_file, sim_tag)

# Loading the similarity matrix
sim_mat = np.loadtxt(sim_file_path, delimiter=";")
# And the corresponding list of types
with open(typefreq_file_path, 'r') as typefreq_file:
    csv_reader = csv.reader(typefreq_file, delimiter=";")
    type_list = [row[0] for row in csv_reader]
# Compute the extended version of the similarity matrix
sim_ext_mat, token_list, existing_index_list = type_to_token_matrix_expansion(text_file_path, sim_mat, type_list)

# Loading ground truth
with open(ground_truth_path) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
real_group_vec = real_group_vec[existing_index_list]

# For semi-supervised results, pick some labels
if known_label_ratio > 0:
    indices_for_known_label = rdm.sample(range(len(real_group_vec)), int(len(real_group_vec) * known_label_ratio))
    known_labels = np.zeros(len(real_group_vec))
    known_labels[indices_for_known_label] = real_group_vec[indices_for_known_label]
    known_labels = known_labels.astype(int)
else:
    known_labels = []
    indices_for_known_label = []

# Compute the dissimilarity matrix
d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                   exch_mat_opt=exch_mat_opt,
                                                   exch_range=exch_range)

# Compute the membership matrix
if known_label_ratio > 0:
    result_matrix = segm_function(d_ext_mat=d_ext_mat,
                                  exch_mat=exch_mat,
                                  w_mat=w_mat,
                                  n_groups=n_groups,
                                  alpha=alpha,
                                  beta=beta,
                                  kappa=kappa,
                                  init_labels=known_labels,
                                  max_it=max_it)
else:
    result_matrix = segm_function(d_ext_mat=d_ext_mat,
                                  exch_mat=exch_mat,
                                  w_mat=w_mat,
                                  n_groups=n_groups,
                                  alpha=alpha,
                                  beta=beta,
                                  kappa=kappa,
                                  max_it=max_it)

# Compute the groups
algo_group_value = np.argmax(result_matrix, 1) + 1

# Compute the real membership matrix
z_real_mat = np.zeros((len(token_list), n_groups))
for i, label in enumerate(real_group_vec):
    if label != 0:
        z_real_mat[i, :] = 0
        z_real_mat[i, label - 1] = 1

# Compute nmi score
nmi = normalized_mutual_info_score(np.delete(real_group_vec, indices_for_known_label),
                                   np.delete(algo_group_value, indices_for_known_label))

# Compute the aggregate labels
df_results = pd.DataFrame(result_matrix)
df_results["Token"] = token_list
type_results = df_results.groupby("Token").mean()
type_list = list(type_results.index)
type_values = type_results.to_numpy()

# -------------------------------------
# --- Writing
# -------------------------------------

# Write html results
write_groups_in_html_file(output_html, token_list, result_matrix)
# Write real html results
write_groups_in_html_file(real_html, token_list, z_real_mat)
# Write csv results
write_membership_mat_in_csv_file(token_csv, token_list, result_matrix)
# Write csv type result
write_membership_mat_in_csv_file(type_csv, type_list, type_values)
# Print nmi
print(nmi)
