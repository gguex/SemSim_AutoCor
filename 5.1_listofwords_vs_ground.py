from local_functions import load_sim_matrix, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, token_clustering, write_groups_in_html_file, write_membership_mat_in_csv_file
import numpy as np
import random as rdm
from sklearn.metrics import normalized_mutual_info_score, average_precision_score
import pandas as pd
from segeval import convert_positions_to_masses, pk, window_diff
from itertools import permutations

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_file = "corpora/manifesto_pp/61320_199211_pp_wostw.txt"
input_group_file = "corpora/manifesto_pp/61320_199211_pp_wostw_groups.txt"
input_sim_file = "similarity_matrices/61320_199211_pp_wostw_w2v.csv"

output_names_root = "results/61320_199211_word_w2v"

n_groups = 7
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 2
beta = 100
kappa = 1

# List of words for each groups
# word_per_groups = [["united", "military", "security", "nuclear", "international", "nations", "peace", "forces",
#                     "allies", "war"],
#                    ["constitution", "constitutional", "vote", "political", "constitutions", "elections", "document",
#                     "separation", "judiciary", "declaration"],
#                    ["local", "state", "washington", "government", "governments", "tribal", "control",
#                     "corruption", "audit", "vision"],
#                    ["tax", "economic", "economy", "energy", "growth", "small", "businesses", "private",
#                     "investment", "trade"],
#                    ["health", "care", "education", "democrats", "students", "schools", "disabilities",
#                     "access", "school", "public"],
#                    ["abortion", "religious", "marriage", "family", "enforcement", "faith", "life", "crime",
#                     "ban", "oppose"],
#                    ["workers", "jobs", "labor", "veterans", "class", "union", "middle", "work",
#                     "farmers", "wages"]]

# List of words for each groups
word_per_groups = [["nuclear", "military", "united", "peace", "international", "allies", "forces", "israel",
                    "security", "weapons"],
                   ["constitution", "rights", "vote", "constitutional", "democracy", "constitutions", "political",
                    "judiciary", "freedom", "amendment"],
                   ["local", "federal", "government", "washington", "state", "audit", "governments",
                    "states", "corruption", "federalism"],
                   ["tax", "growth", "economy", "economic", "businesses", "energy", "innovation", "trade",
                    "market", "investment"],
                   ["health", "care", "education", "disabilities", "students", "schools", "school",
                    "access", "color", "mental"],
                   ["abortion", "marriage", "religious", "crime", "faith", "family", "enforcement", "life",
                    "beliefs", "intelligence"],
                   ["workers", "jobs", "labor", "class", "union", "veterans", "unions", "organize",
                    "wages", "farmers"]]

# List of words for each groups
# word_per_groups = [["nuclear", "military", "united", "security", "nato", "international", "israel", "nations", "forces",
#                     "russia"],
#                    ["rights", "constitution", "democracy", "political", "freedom", "vote", "people", "civil",
#                     "amendment", "citizens"],
#                    ["federal", "government", "local", "state", "washington", "administration", "tribal", "republican",
#                     "congress", "party"],
#                    ["tax", "economic", "economy", "energy", "businesses", "growth", "private", "market", "trade",
#                     "companies"],
#                    ["health", "education", "care", "students", "disabilities", "schools", "democrats", "access",
#                     "public", "colors"],
#                    ["abortion", "marriage", "religious", "family", "enforcement", "life", "crime", "first",
#                     "children", "life"],
#                    ["class", "labor", "farmers", "workers", "union", "wages", "veterans", "jobs", "middle", "working"]]

# -------------------------------------
# --- Computations
# -------------------------------------

# Loading the similarity matrix
type_list, sim_mat = load_sim_matrix(input_sim_file)

# Compute the extended version of the similarity matrix
sim_ext_mat, token_list, existing_index_list = type_to_token_matrix_expansion(input_text_file, sim_mat, type_list)

# Loading ground truth
with open(input_group_file) as ground_truth:
    real_group_vec = ground_truth.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])
real_group_vec = real_group_vec[existing_index_list]

# Label the words
known_labels = np.zeros(len(real_group_vec))
indices_for_known_label = []
for i in range(len(word_per_groups)):
    group_words = word_per_groups[i]
    indices_group_words = [token_id for token_id, token in enumerate(token_list) if token in group_words]
    indices_for_known_label.extend(indices_group_words)
    known_labels[indices_group_words] = (i+1)
known_labels = known_labels.astype(int)

# Some stats correct percentage
nb_in_groups_list = []
nb_right_groups_list = []
for i in range(n_groups):
    nb_in_groups = sum(known_labels == i+1)
    nb_right_groups = sum(real_group_vec[known_labels == i+1] == i+1)
    print(f"Groupe {i+1}: Nb = {nb_in_groups} Percent_right = {nb_right_groups / nb_in_groups}")
    nb_in_groups_list.append(nb_in_groups)
    nb_right_groups_list.append(nb_right_groups)

print(f"Overall : Nb = {sum(nb_in_groups_list)} ({sum(nb_in_groups_list) / len(token_list)}%) "
      f"Percent_right = {sum(nb_right_groups_list) / sum(nb_in_groups_list)}")

# Compute the dissimilarity matrix
d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

# Compute the exchange and transition matrices
exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                   exch_mat_opt=exch_mat_opt,
                                                   exch_range=exch_range)

# Compute the membership matrix
result_matrix = token_clustering(d_ext_mat=d_ext_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups, alpha=alpha,
                                 beta=beta, kappa=kappa, init_labels=known_labels, verbose=True)

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
write_groups_in_html_file(output_names_root + "_aglo.html", token_list, result_matrix,
                          comment_line=f"nmi = {nmi}, pk={pk_res}, win_diff={win_diff}")
# Write real html results
write_groups_in_html_file(output_names_root + "_real.html", token_list, z_real_mat, comment_line="Real results")
# Write csv results
write_membership_mat_in_csv_file(output_names_root + "_token.csv", token_list, result_matrix)
# Write csv type result
write_membership_mat_in_csv_file(output_names_root + "_type.csv", type_list, type_values)
# Print nmi, pk_res, win_diff
print(f"nmi = {nmi}, map = {map}, pk={pk_res} (rdm={pk_rdm}), win_diff={win_diff} (rdm={win_diff_rdm})")
