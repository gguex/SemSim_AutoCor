from code_python.local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, discontinuity_segmentation, cut_segmentation
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score
from itertools import compress

# -------------------------------------
# --- Parameters
# -------------------------------------

# Segmentation tag ("disc" or "cut")
segm_tag = "cut"

# Number of crossval folds
n_fold = 5

# --- Experiments loop lists (to make several experiments)

# List of inputted text files to explore
input_file_list = ["mix_sent1.txt"]
# List of names for the ouputted result files
results_file_name_list = ["results_crossval5_sent1_cut_1.csv"]
# List of label ratios to text
known_label_ratio_list = [0]
# List of similarity tag
sim_tag_list = ["w2v"]
# List of number of groups
n_groups_list = [4]
# Dist options to explore
dist_option_list = ["minus_log"]

# --- Grid search parameters

exch_mat_opt_vec = ["d"]
exch_range_vec = [3, 5, 10, 15]
alpha_vec = [0.1, 1, 2, 5, 10, 50, 100]
beta_vec = [0.1, 1, 5, 10, 50, 100, 300]
kappa_vec = [0, 1 / 3, 2 / 3, 1]

# -------------------------------------
# --- Computations
# -------------------------------------

# Selection of the segmentation function
if segm_tag == "disc":
    segm_function = discontinuity_segmentation
else:
    segm_function = cut_segmentation

for i in range(len(input_file_list)):

    # -------------------------------------
    # --- Parameters
    # -------------------------------------

    # File name to explore
    input_file = input_file_list[i]
    # File name to save
    results_file_name = results_file_name_list[i]
    # Ratio of known labels. If 0, clustering
    known_label_ratio = known_label_ratio_list[i]
    # Similarity tag
    sim_tag = sim_tag_list[i]
    # Number of groups
    n_groups = n_groups_list[i]
    # Dist option
    dist_option = dist_option_list[i]

    # Print
    print(f"Crossval on {input_file}, known ratio = {known_label_ratio}, sim tag = {sim_tag}, n_groups = {n_groups}, "
          f"dist option = {dist_option}")

    # Make results file
    with open(results_file_name, "w") as output_file:
        output_file.write("input_file,known_label_ratio,sim_tag,n_groups,dist_option,fold_id,exch_mat_opt,exch_range,"
                          "alpha,beta,kappa,nmi_train,nmi_test\n")

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
    # Compute the dissimilarity matrix
    d_ext_mat = similarity_to_dissimilarity(sim_ext_mat, dist_option=dist_option)

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
        train_d_mat = d_ext_mat[train_id, :][:, train_id]
        train_real_group_vec = real_group_vec[train_id]

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(train_real_group_vec)),
                                                 int(len(train_real_group_vec) * known_label_ratio))
            known_labels = np.zeros(len(train_real_group_vec))
            known_labels[indices_for_known_label] = train_real_group_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = []
            indices_for_known_label = []

        # ----- TRAIN

        # Compute best parameters
        nmi_train = 0
        best_param_dic = {}
        for exch_mat_opt in exch_mat_opt_vec:
            for exch_range in exch_range_vec:

                # Compute the exchange and transition matrices
                train_exch_mat, train_w_mat = exchange_and_transition_matrices(len(train_token_list),
                                                                               exch_mat_opt=exch_mat_opt,
                                                                               exch_range=exch_range)

                for alpha in alpha_vec:
                    for beta in beta_vec:
                        for kappa in kappa_vec:
                            # Compute the matrix
                            if known_label_ratio > 0:
                                result_matrix = segm_function(d_ext_mat=train_d_mat,
                                                              exch_mat=train_exch_mat,
                                                              w_mat=train_w_mat,
                                                              n_groups=4,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              kappa=kappa,
                                                              init_labels=known_labels)
                            else:
                                result_matrix = segm_function(d_ext_mat=train_d_mat,
                                                              exch_mat=train_exch_mat,
                                                              w_mat=train_w_mat,
                                                              n_groups=4,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              kappa=kappa)

                            # Compute the groups
                            algo_group_value = np.argmax(result_matrix, 1) + 1

                            # Compute nmi score
                            nmi = normalized_mutual_info_score(np.delete(train_real_group_vec, indices_for_known_label),
                                                               np.delete(algo_group_value, indices_for_known_label))

                            # If nmi is better, write it
                            if nmi > nmi_train:
                                nmi_train = nmi
                                best_param_dic = {"exch_mat_opt": exch_mat_opt,
                                                  "exch_range": exch_range,
                                                  "alpha": alpha,
                                                  "beta": beta,
                                                  "kappa": kappa}
                                print(f"New best: {nmi_train}, {best_param_dic}")

        # ----- TEST

        # Setting test id and restricting to test set
        test_id = crossval_index != fold_id
        test_token_list = list(compress(token_list, test_id))
        test_d_mat = d_ext_mat[test_id, :][:, test_id]
        test_real_group_vec = real_group_vec[test_id]

        # For semi-supervised results, pick some labels
        if known_label_ratio > 0:
            indices_for_known_label = rdm.sample(range(len(test_real_group_vec)),
                                                 int(len(test_real_group_vec) * known_label_ratio))
            known_labels = np.zeros(len(test_real_group_vec))
            known_labels[indices_for_known_label] = test_real_group_vec[indices_for_known_label]
            known_labels = known_labels.astype(int)
        else:
            known_labels = []
            indices_for_known_label = []

        # Compute the exchange and transition matrices
        test_exch_mat, test_w_mat = exchange_and_transition_matrices(len(test_token_list),
                                                                     exch_mat_opt=best_param_dic["exch_mat_opt"],
                                                                     exch_range=best_param_dic["exch_range"])

        # Compute the matrix
        if known_label_ratio > 0:
            result_matrix = segm_function(d_ext_mat=test_d_mat,
                                          exch_mat=test_exch_mat,
                                          w_mat=test_w_mat,
                                          n_groups=4,
                                          alpha=best_param_dic["alpha"],
                                          beta=best_param_dic["beta"],
                                          kappa=best_param_dic["kappa"],
                                          init_labels=known_labels)
        else:
            result_matrix = segm_function(d_ext_mat=test_d_mat,
                                          exch_mat=test_exch_mat,
                                          w_mat=test_w_mat,
                                          n_groups=4,
                                          alpha=best_param_dic["alpha"],
                                          beta=best_param_dic["beta"],
                                          kappa=best_param_dic["kappa"])

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
            output_file.write(f"{input_file},{known_label_ratio},{sim_tag},{n_groups},{dist_option},{fold_id},"
                              f"{exch_mat_opt},{exch_range},{alpha},{beta},{kappa},{nmi_train},{nmi_test}\n")
