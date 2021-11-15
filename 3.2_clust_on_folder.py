from local_functions import *
import numpy as np
import random as rdm
from sklearn.metrics import normalized_mutual_info_score, average_precision_score
from itertools import permutations

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_folder = "corpora/manifesto_pp"
stop_words = False
output_file = "results/clust_manifesto.csv"

#---

sim_tag = "w2v"
dist_option = "max_minus"
exch_mat_opt = "u"
exch_range = 15
alpha = 2
beta = 100
kappa = 1
known_label_ratio = 0  # if > 0, semi-supervised model

n_tests = 10

# -------------------------------------
# --- Computations
# -------------------------------------

file_list = os.listdir(input_text_folder)

# Restrict them to those with or without stopwords
file_list = [file for file in file_list if ("wostw" in file) ^ stop_words]

# Sort the list
file_list.sort()

# Split groups and non-groups file
text_file_list = [file for file in file_list if "groups" not in file]
input_text_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" not in file]
input_group_file_list = [f"{input_text_folder}/{file}" for file in file_list if "groups" in file]
input_sim_file_list = [f"similarity_matrices/{file[:-4]}_{sim_tag}.csv" for file in file_list if "groups" not in file]

with open(output_file, "w") as res_file:
    res_file.write(f"file,nmi,map,pk,win_diff,pk_rdm,win_diff_rdm\n")

for index_file in range(len(input_text_file_list)):

    input_text_file = input_text_file_list[index_file]
    input_group_file = input_group_file_list[index_file]
    input_sim_file = input_sim_file_list[index_file]

    # Loading the similarity matrix
    type_list, sim_mat = load_sim_matrix(input_sim_file)

    # Compute the extended version of the similarity matrix
    sim_ext_mat, token_list, existing_index_list = type_to_token_matrix_expansion(input_text_file, sim_mat, type_list)

    # Loading ground truth
    with open(input_group_file) as ground_truth:
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

    nmi_vec = []
    map_vec = []
    pk_vec = []
    win_diff_vec = []
    pk_rdm_vec = []
    win_diff_rdm_vec = []
    for id_test in range(n_tests):

        # Compute the membership matrix
        result_matrix = token_clustering(d_ext_mat=d_ext_mat, exch_mat=exch_mat, w_mat=w_mat, n_groups=n_groups, alpha=alpha,
                                         beta=beta, kappa=kappa, known_labels=known_labels, verbose=True)

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

        # Compute nmi scorec
        nmi = normalized_mutual_info_score(rstr_real_group_vec, rstr_algo_group_vec)
        # Compute Map
        ap_vector = [average_precision_score(rstr_best_real_group_vec == group_id, rstr_algo_group_vec == group_id)
                     for group_id in range(1, max(rstr_real_group_vec) + 1)]
        map = np.mean(ap_vector)

        # Segmentation evaluation
        pk_res, win_diff, pk_rdm, win_diff_rdm = seg_eval(algo_group_vec, real_group_vec)

        nmi_vec.append(nmi)
        map_vec.append(map)
        pk_vec.append(pk_res)
        win_diff_vec.append(win_diff)
        pk_rdm_vec.append(pk_rdm)
        win_diff_rdm_vec.append(win_diff_rdm)


    with open(output_file, "a") as res_file:
        res_file.write(f"{input_text_file},{np.mean(nmi_vec)},{np.mean(map_vec)},{np.mean(pk_vec)},"
                       f"{np.mean(win_diff_vec)},{np.mean(pk_rdm_vec)},{np.mean(win_diff_rdm_vec)}\n")
