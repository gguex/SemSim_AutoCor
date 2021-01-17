from code_python.local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, discontinuity_segmentation, cut_segmentation
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
from itertools import compress, product

# -------------------------------------
# --- Parameters
# -------------------------------------

# Segmentation tag ("disc" or "cut")
segm_tag = "cut"

# Number of crossval folds
n_fold = 5

# Number of tests on each fold
n_test = 5

# List of names for the ouputted result files
results_file_name = "grid_search_results/cv5_all_2.csv"

# --- Experiments loop lists (to make several experiments)

# List of inputted text files to explore
input_file_list = ["mix_sent10.txt", "mix_sent1.txt", "mix_word3.txt", "mix_word1.txt",
                   "mix_sent10.txt", "mix_sent1.txt", "mix_word3.txt", "mix_word1.txt"]
# List of label ratios to text
known_label_ratio_list = [0, 0, 0, 0,
                          0, 0, 0, 0]
# List of similarity tag
sim_tag_list = ["path", "path", "path", "path",
                "wup", "wup", "wup", "wup"]
# List of number of groups
n_groups_list = [4, 4, 4, 4,
                 4, 4, 4, 4]

# --- Grid search parameters

dist_option_vec = ["max_minus"]
exch_mat_opt_vec = ["s", "u", "d"]
exch_range_vec = [3, 5, 10, 15]
alpha_vec = [0.1, 1, 2, 5, 10, 30]
beta_vec = [5, 10, 50, 100, 200]
kappa_vec = [0, 0.25, 0.5, 0.75, 1]

# -------------------------------------
# --- Computations
# -------------------------------------

# Selection of the segmentation function
if segm_tag == "disc":
    segm_function = discontinuity_segmentation
else:
    segm_function = cut_segmentation

# Make results file
with open(results_file_name, "w") as output_file:
    output_file.write("input_file,known_label_ratio,sim_tag,n_groups,fold_id,dist_option,exch_mat_opt,exch_range,"
                      "alpha,beta,kappa,nmi_train,nmi_test\n")

for i in range(len(input_file_list)):

    # -------------------------------------
    # --- Parameters
    # -------------------------------------

    # File name to explore
    input_file = input_file_list[i]
    # Ratio of known labels. If 0, clustering
    known_label_ratio = known_label_ratio_list[i]
    # Similarity tag
    sim_tag = sim_tag_list[i]
    # Number of groups
    n_groups = n_groups_list[i]

    # Print
    print(f"Crossval on {input_file}, known ratio = {known_label_ratio}, sim tag = {sim_tag}, n_groups = {n_groups}")

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

    # Setting crossval groups:
    fold_size = len(token_list) // n_fold
    crossval_index = np.append(np.repeat(range(n_fold), fold_size), np.repeat(n_fold - 1, len(token_list) % n_fold))

    # Cross validation
    for fold_id in range(n_fold):

        # Setting train id and restricting to train set
        train_id = crossval_index == fold_id
        train_token_list = list(compress(token_list, train_id))
        train_s_mat = sim_ext_mat[train_id, :][:, train_id]
        train_real_group_vec = real_group_vec[train_id]

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(train_real_group_vec)),
                                                 int(len(train_real_group_vec) * known_label_ratio))
            known_labels = np.zeros(len(train_real_group_vec))
            known_labels[indices_for_known_label] = train_real_group_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = None
            indices_for_known_label = []

        # ----- TRAIN

        # Compute best parameters
        nmi_train = 0
        best_param_dic = {}
        for dist_option in dist_option_vec:

            # Compute the dissimilarity matrix
            train_d_mat = similarity_to_dissimilarity(train_s_mat, dist_option=dist_option)

            for exch_mat_opt, exch_range in product(exch_mat_opt_vec, exch_range_vec):

                # Compute the exchange and transition matrices
                train_exch_mat, train_w_mat = exchange_and_transition_matrices(len(train_token_list),
                                                                               exch_mat_opt=exch_mat_opt,
                                                                               exch_range=exch_range)

                for alpha, beta, kappa in product(alpha_vec, beta_vec, kappa_vec):

                    # Compute the matrix and nmi  n_test time
                    nmi_vector = []
                    for _ in range(n_test):

                        result_matrix = segm_function(d_ext_mat=train_d_mat,
                                                      exch_mat=train_exch_mat,
                                                      w_mat=train_w_mat,
                                                      n_groups=4,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      kappa=kappa,
                                                      init_labels=known_labels)

                        # Compute the groups
                        algo_group_value = np.argmax(result_matrix, 1) + 1

                        # Compute nmi score
                        nmi_vector.append(normalized_mutual_info_score(np.delete(train_real_group_vec,
                                                                                 indices_for_known_label),
                                                                       np.delete(algo_group_value,
                                                                                 indices_for_known_label)))
                    # Compute mean nmi
                    mean_nmi = np.mean(nmi_vector)
                    # If nmi is better, write it
                    if mean_nmi > nmi_train:
                        nmi_train = mean_nmi
                        best_param_dic = {"dist_option": dist_option,
                                          "exch_mat_opt": exch_mat_opt,
                                          "exch_range": exch_range,
                                          "alpha": alpha,
                                          "beta": beta,
                                          "kappa": kappa}
                        print(f"New best: {nmi_train}, {best_param_dic}")

        # ----- TEST

        # Setting test id and restricting to test set
        test_id = crossval_index != fold_id
        test_token_list = list(compress(token_list, test_id))
        test_s_mat = sim_ext_mat[test_id, :][:, test_id]
        test_real_group_vec = real_group_vec[test_id]

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(test_real_group_vec)),
                                                 int(len(test_real_group_vec) * known_label_ratio))
            known_labels = np.zeros(len(test_real_group_vec))
            known_labels[indices_for_known_label] = test_real_group_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = None
            indices_for_known_label = []

        # Compute the dissimilarity matrix
        test_d_mat = similarity_to_dissimilarity(test_s_mat, dist_option=best_param_dic["dist_option"])

        # Compute the exchange and transition matrices
        test_exch_mat, test_w_mat = exchange_and_transition_matrices(len(test_token_list),
                                                                     exch_mat_opt=best_param_dic["exch_mat_opt"],
                                                                     exch_range=best_param_dic["exch_range"])

        # Compute the matrix
        result_matrix = segm_function(d_ext_mat=test_d_mat,
                                      exch_mat=test_exch_mat,
                                      w_mat=test_w_mat,
                                      n_groups=4,
                                      alpha=best_param_dic["alpha"],
                                      beta=best_param_dic["beta"],
                                      kappa=best_param_dic["kappa"],
                                      init_labels=known_labels)

        # Compute the groups
        algo_group_value = np.argmax(result_matrix, 1) + 1

        # Compute nmi score
        nmi_test = normalized_mutual_info_score(np.delete(test_real_group_vec, indices_for_known_label),
                                                np.delete(algo_group_value, indices_for_known_label))

        # Printing best param and nmi
        print(f"Fold {fold_id + 1}/{n_fold} : nmi train = {nmi_train}, nmi test = {nmi_test}, "
              f"best param = {best_param_dic}")

        # Writing results
        with open(results_file_name, "a") as output_file:
            output_file.write(f"{input_file},{known_label_ratio},{sim_tag},{n_groups},{fold_id},"
                              f"{best_param_dic['dist_option']},{best_param_dic['exch_mat_opt']},"
                              f"{best_param_dic['exch_range']},{best_param_dic['alpha']},{best_param_dic['beta']},"
                              f"{best_param_dic['kappa']},{nmi_train},{nmi_test}\n")
