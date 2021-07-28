from local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, cut_clustering
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# input_file_list = ["61320_199211_pp.txt",
#                    "61320_200411_pp.txt",
#                    "61320_201211_pp.txt",
#                    "61320_201611_pp.txt",
#                    "61620_200411_pp.txt",
#                    "61620_200811_pp.txt",
#                    "61620_201211_pp.txt",
#                    "61620_201611_pp.txt"]
#
# sim_tag_list = ["w2v"] * len(input_file_list)
# dist_option_list = ["max_minus"] * len(input_file_list)
# exch_mat_opt_list = ["u"] * len(input_file_list)
# exch_range_list = [15] * len(input_file_list)
# alpha_list = [5] * len(input_file_list)
# beta_list = [50] * len(input_file_list)
# kappa_list = [0.5] * len(input_file_list)
# known_label_ratio_list = [0] * len(input_file_list)  # if 0, clustering
# n_test_list = [5] * len(input_file_list)
#
# results_file_name = "../results/5_clust_results/5_manifesto_multiclust_test.csv"

exch_range_list = [10] * (4 * 4 * 4) + [15] * (4 * 4 * 4)
alpha_list = ([2] * (4 * 4) + [5] * (4 * 4) + [10] * (4 * 4) + [30] * (4 * 4)) * 2
beta_list = (([10] * 4 + [50] * 4 + [100] * 4 + [200] * 4) * 4) * 2
kappa_list = (([0, 0.33, 0.66, 1] * 4) * 4) * 2

input_file_list = ["61320_201211_pp.txt"] * len(exch_range_list)

sim_tag_list = ["w2v"] * len(input_file_list)
dist_option_list = ["max_minus"] * len(input_file_list)
exch_mat_opt_list = ["u"] * len(input_file_list)
known_label_ratio_list = [0] * len(input_file_list)  # if 0, clustering
n_test_list = [5] * len(input_file_list)

results_file_name = "../results/5_clust_results/5_manifesto_param_search3.csv"

# -------------------------------------
# --- Computations
# -------------------------------------

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("input_file,sim_tag,dist_opt,exch_opt,exch_range,alpha,beta,kappa,known_label,n_test,"
                      "mean_nmi,ci95_nmi\n")

# Loop on file
for i, input_file in enumerate(input_file_list):

    # Get hyperparameters
    sim_tag = sim_tag_list[i]
    dist_option = dist_option_list[i]
    exch_mat_opt = exch_mat_opt_list[i]
    exch_range = exch_range_list[i]
    alpha = alpha_list[i]
    beta = beta_list[i]
    kappa = kappa_list[i]
    known_label_ratio = known_label_ratio_list[i]
    n_test = n_test_list[i]

    # Print
    print(f"File: {input_file}, {sim_tag}, {dist_option}, {exch_mat_opt}, {exch_range}, {alpha}, {beta}, {kappa},"
          f" {known_label_ratio}, {n_test}")

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
    n_group = len(set(real_group_vec))

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
    for _ in tqdm(range(n_test)):
        # Compute the membership matrix
        result_matrix = cut_clustering(d_ext_mat=d_ext_mat,
                                       exch_mat=exch_mat,
                                       w_mat=w_mat,
                                       n_group=n_group,
                                       alpha=alpha,
                                       beta=beta,
                                       kappa=kappa,
                                       init_labels=known_labels)

        # Compute the groups
        algo_group_vec = np.argmax(result_matrix, 1) + 1

        # Restrained results
        rstr_real_group_vec = np.delete(real_group_vec, indices_for_known_label)
        rstr_algo_group_vec = np.delete(algo_group_vec, indices_for_known_label)

        # Compute nmi score
        nmi_vec.append(normalized_mutual_info_score(rstr_real_group_vec, rstr_algo_group_vec))

    # Writing results
    nmi_mean = np.mean(nmi_vec)
    nmi_std = np.std(nmi_vec)
    with open(results_file_name, "a") as output_file:
        output_file.write(f"{input_file},{sim_tag},{dist_option},{exch_mat_opt},{exch_range},{alpha},{beta},{kappa},"
                          f"{known_label_ratio},{n_test},{nmi_mean},{nmi_std * 1.96 / np.sqrt(n_test)}\n")
