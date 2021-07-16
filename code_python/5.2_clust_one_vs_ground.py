from local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, discontinuity_clustering, cut_clustering, \
    write_groups_in_html_file, write_membership_mat_in_csv_file
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score, average_precision_score
import pandas as pd
from segeval import convert_positions_to_masses, pk, window_diff
from itertools import permutations

# -------------------------------------
# --- Parameters
# -------------------------------------

input_file = "61320_200411_pp.txt"

output_html = "best_rep2004.html"
real_html = "real_rep2004.html"
token_csv = "best_rep2004_tokens.csv"
type_csv = "best_rep2004_types.csv"

sim_tag = "ftx"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 5
beta = 50
kappa = 0.5
known_label_ratio = 0  # if 0, clustering
clust_tag = "cut"  # Clustering method tag ("disc" or "cut")
max_it = 200

# -------------------------------------
# --- Computations
# -------------------------------------

# Selection of the clustering function
if clust_tag == "disc":
    clust_function = discontinuity_clustering
else:
    clust_function = cut_clustering

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
n_groups = len(set(real_group_vec))

# For semi-supervised results, pick some labels
if known_label_ratio > 0:
    indices_for_known_label = rdm.sample(range(len(real_group_vec)), int(len(real_group_vec) * known_label_ratio))
    known_labels = np.zeros(len(real_group_vec))
    known_labels[indices_for_known_label] = real_group_vec[indices_for_known_label]
    known_labels = known_labels.astype(int)
else:
    known_labels = None
    indices_for_known_label = []

# Compute the dissimilarity matrix
d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                   exch_mat_opt=exch_mat_opt,
                                                   exch_range=exch_range)

# Compute the membership matrix
result_matrix = clust_function(d_ext_mat=d_ext_mat,
                               exch_mat=exch_mat,
                               w_mat=w_mat,
                               n_groups=n_groups,
                               alpha=alpha,
                               beta=beta,
                               kappa=kappa,
                               init_labels=known_labels,
                               max_it=max_it,
                               verbose=True)

# Compute the groups
algo_group_vec = np.argmax(result_matrix, 1) + 1

# Permutation of real group (for most matching colors)
original_group = list(range(1, n_groups + 1))
best_real_group_vec = real_group_vec
best_nb_match = np.sum(real_group_vec == algo_group_vec)
for perm in list(permutations(original_group)):
    perm = np.array(perm)
    test_real_group_vec = perm[real_group_vec - 1]
    test_nb_match = np.sum(test_real_group_vec == algo_group_vec)
    if test_nb_match > best_nb_match:
        best_nb_match = test_nb_match
        best_real_group_vec = test_real_group_vec

# Compute the real membership matrix
z_real_mat = np.zeros((len(token_list), n_groups))
for i, label in enumerate(best_real_group_vec):
    if label != 0:
        z_real_mat[i, :] = 0
        z_real_mat[i, label - 1] = 1

# Restrained results
rstr_real_group_vec = np.delete(real_group_vec, indices_for_known_label)
rstr_algo_group_vec = np.delete(algo_group_vec, indices_for_known_label)
rstr_best_real_group_vec = np.delete(best_real_group_vec, indices_for_known_label)

# Compute nmi score
nmi = normalized_mutual_info_score(rstr_real_group_vec, rstr_algo_group_vec)
# Compute Map
ap_vector = [average_precision_score(rstr_best_real_group_vec == group_id, rstr_algo_group_vec == group_id)
             for group_id in range(1, max(rstr_real_group_vec) + 1)]
map = np.mean(ap_vector)

# Segmentation evaluation
real_segm_vec = convert_positions_to_masses(rstr_real_group_vec)
algo_segm_vec = convert_positions_to_masses(rstr_algo_group_vec)
rdm_group_vec = rstr_real_group_vec.copy()
rdm.shuffle(rdm_group_vec)
rdm_segm_vec = convert_positions_to_masses(rdm_group_vec)
pk_res = pk(algo_segm_vec, real_segm_vec)
win_diff = window_diff(algo_segm_vec, real_segm_vec)
pk_rdm = pk(rdm_segm_vec, real_segm_vec)
win_diff_rdm = window_diff(rdm_segm_vec, real_segm_vec)

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
write_groups_in_html_file(output_html, token_list, result_matrix, comment_line=f"nmi = {nmi}, pk={pk_res}, "
                                                                               f"win_diff={win_diff}")
# Write real html results
write_groups_in_html_file(real_html, token_list, z_real_mat, comment_line="Real results")
# Write csv results
write_membership_mat_in_csv_file(token_csv, token_list, result_matrix)
# Write csv type result
write_membership_mat_in_csv_file(type_csv, type_list, type_values)
# Print nmi, pk_res, win_diff
print(f"nmi = {nmi}, map = {map}, pk={pk_res} (rdm={pk_rdm}), win_diff={win_diff} (rdm={win_diff_rdm})")
