from local_functions import load_sim_matrix, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, token_clustering
import numpy as np
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
from itertools import product
import multiprocessing as mp
from miniutils import parallel_progbar

# -------------------------------------
# --- Parameters
# -------------------------------------

input_text_file = "corpora/manifesto_pp/61620_200411_pp_wostw.txt"
input_group_file = "corpora/manifesto_pp/61620_200411_pp_wostw_groups.txt"

results_file_name = "results/search_big_200411.csv"

# Known label ?
known_label_ratio = 0

# Number of tests
n_tests = 4

# Number of groups
n_groups = 7

# Search on
input_sim_file_list = ["similarity_matrices/61620_200411_pp_wostw_w2v.csv",
                       "similarity_matrices/61620_200411_pp_wostw_glv.csv",
                       "similarity_matrices/61620_200411_pp_wostw_ftx.csv"]
dist_option_vec = ["max_minus"]
exch_mat_opt_vec = ["u", "d"]
exch_range_vec = [5, 10, 15]
alpha_vec = [1, 2, 5, 10, 30]
beta_vec = [5, 10, 50, 100, 200]
kappa_vec = [0, 0.25, 0.5, 0.75, 1]

# Number of cpu to use
n_cpu = mp.cpu_count()

# -------------------------------------
# --- Computations
# -------------------------------------

# Creating hyperparameters for multiproc
hyperp_list = list(product(alpha_vec, beta_vec, kappa_vec))

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("input_file,label_ratio,sim_tag,n_groups,n_tests,dist_option,exch_mat_opt,exch_range,"
                      "alpha,beta,kappa,mean_nmi\n")

######################
#### Loop on sim files

for input_sim_file in input_sim_file_list:

    # Get sim tag
    sim_tag = input_sim_file[-7:-4]

    # Loading the similarity matrix
    type_list, sim_mat = load_sim_matrix(input_sim_file)

    # Compute the extended version of the similarity matrix
    sim_ext_mat, token_list, existing_index_list = type_to_token_matrix_expansion(input_text_file, sim_mat, type_list)

    # Loading ground truth
    with open(input_group_file) as ground_truth:
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
        known_labels = None
        indices_for_known_label = []

    # Restrained real label
    rstr_real_group_vec = np.delete(real_group_vec, indices_for_known_label)

    ########################
    #### Loop on dist option

    for dist_option in dist_option_vec:

        # Compute the dissimilarity matrix
        d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

        ######################
        #### Loop on exchg opt

        for exch_mat_opt, exch_range in product(exch_mat_opt_vec, exch_range_vec):

            # Compute the exchange and transition matrices
            exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                               exch_mat_opt=exch_mat_opt,
                                                               exch_range=exch_range)


            ########################################
            #### Creating a function to multiprocess

            def nmi_computation(alpha, beta, kappa):
                # Compute the matrix and nmi  n_train time
                nmi_vector = []
                for _ in range(n_tests):
                    # Compute the membership matrix
                    res_matrix = token_clustering(d_ext_mat=d_ext_mat,
                                                  exch_mat=exch_mat,
                                                  w_mat=w_mat,
                                                  n_groups=n_groups,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  kappa=kappa,
                                                  known_labels=known_labels)
                    # Compute the groups
                    alg_group_vec = np.argmax(res_matrix, 1) + 1
                    rstr_alg_group_vec = np.delete(alg_group_vec, indices_for_known_label)
                    # Compute nmi score
                    nmi = normalized_mutual_info_score(rstr_real_group_vec, rstr_alg_group_vec)
                    nmi_vector.append(nmi)

                return np.mean(nmi_vector)

            ##################################
            #### Computing and writing results

            # Print message
            print(f"Multiprocessing for {sim_tag}, {dist_option}, {exch_mat_opt}, {exch_range}")
            # Multiprocess
            res_multi = parallel_progbar(nmi_computation, hyperp_list, starmap=True, nprocs=n_cpu)

            # Writing results
            with open(results_file_name, "a") as output_file:
                for id_hyp, hyperp in enumerate(hyperp_list):
                    output_file.write(f"{input_text_file},{known_label_ratio},{sim_tag},{n_groups},{n_tests},"
                                      f"{dist_option},{exch_mat_opt},{exch_range},{hyperp[0]},{hyperp[1]},"
                                      f"{hyperp[2]},{res_multi[id_hyp]}\n")
