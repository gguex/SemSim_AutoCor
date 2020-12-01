from code_python.local_functions import sim_to_dissim_matrix, exchange_and_transition_matrices, \
    discontinuity_segmentation
from sklearn.metrics import normalized_mutual_info_score
import numpy as np

# import random as rdm
# from sklearn.metrics import confusion_matrix

# --- PARAMETERS

# File name to save
results_file_name = "results_sent1_big_1.csv"

# Dist options to explore
dist_option_vec = ["minus_log", "1_minus"]

# Exchange matrix options to explore
exch_mat_opt_vec = ["s", "u", "d"]
exch_range_vec = [1, 3, 5, 10]

# Parameter values to explore
alpha_vec = [1, 5, 10, 50, 100, 500]
beta_vec = [1, 5, 10, 50, 100, 500]
kappa_vec = [0, 1 / 3, 2 / 3, 1]

# --- GRID SEARCH

# Ground truth
with open("/home/gguex/PycharmProjects/SemSim_AutoCor/corpora/mixgroup_sent1_min5.txt") as group_file:
    real_group_vec = group_file.read()
    real_group_vec = np.array([int(element) for element in real_group_vec.split(",")])

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("dist,exch_opt,exch_range,alpha,beta,kappa,nmi\n")

print("Starting")

for dist_option in dist_option_vec:
    # Compute the dissimilartiy_matrix
    d_ext_mat = sim_to_dissim_matrix(input_file="mix_sent1_min5.txt",
                                     sim_tag="wesim",
                                     dist_option=dist_option)

    print(f"Dissimilarity matrix computed with {dist_option}")

    for exch_mat_opt in exch_mat_opt_vec:
        for exch_range in exch_range_vec:

            # Compute the exchange and transition matrices
            exch_mat, w_mat = exchange_and_transition_matrices(input_file="mix_sent1_min5.txt", sim_tag="wesim",
                                                               exch_mat_opt=exch_mat_opt, exch_range=exch_range)

            print(f"Exchange matrix computed with {exch_mat_opt} and range {exch_range}")

            for alpha in alpha_vec:
                for beta in beta_vec:
                    for kappa in kappa_vec:
                        # Compute the matrix
                        result_matrix = discontinuity_segmentation(d_ext_mat=d_ext_mat,
                                                                   exch_mat=exch_mat,
                                                                   w_mat=w_mat,
                                                                   n_groups=4,
                                                                   alpha=alpha,
                                                                   beta=beta,
                                                                   kappa=kappa)

                        # Compute the groups
                        algo_group_value = np.argmax(result_matrix, 1) + 1

                        # Compute nmi score
                        nmi = normalized_mutual_info_score(real_group_vec, algo_group_value)
                        print(f"NMI = {nmi}")

                        # Writing results
                        with open(results_file_name, "a") as output_file:
                            output_file.write(f"{dist_option},{exch_mat_opt},{exch_range},{alpha},{beta},{kappa},"
                                              f"{nmi}\n")

# ---- WITH LABEL

# Give some label
# percent_of_know_label = 0.1
# index_to_keep = rdm.sample(range(len(real_group_vec)), int(len(real_group_vec) * percent_of_know_label))
# known_labels = np.zeros(len(real_group_vec))
# known_labels[index_to_keep] = real_group_vec[index_to_keep]
# known_labels = known_labels.astype(int)
#
# # Compute the results matrix
# result_matrix = compute_discontinuity_segment_token(d_ext_mat=d_ext_mat,
#                                                     exch_mat=exch_mat,
#                                                     w_mat=w_mat,
#                                                     n_groups=4,
#                                                     alpha=3,
#                                                     beta=10,
#                                                     kappa=0.8,
#                                                     init_labels=known_labels)
#
# # Getting the group attribution (crisp)
# algo_group_value = np.argmax(result_matrix, 1) + 1
#
# # Comparing with ground truth
# #conf_matrix = confusion_matrix(real_group_vec, algo_group_value)
# #nmi = normalized_mutual_info_score(real_group_vec, algo_group_value)
# conf_matrix = confusion_matrix(np.delete(real_group_vec, index_to_keep), np.delete(algo_group_value, index_to_keep))
# nmi = normalized_mutual_info_score(np.delete(real_group_vec, index_to_keep), \
#     np.delete(algo_group_value, index_to_keep))
# print(conf_matrix)
# print(f"NMI = {nmi}")
