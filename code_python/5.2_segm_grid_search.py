from code_python.local_functions import get_all_paths, type_to_token_matrix_expansion, similarity_to_dissimilarity, \
    exchange_and_transition_matrices, discontinuity_segmentation, cut_segmentation
import numpy as np
import csv
import random as rdm
from sklearn.metrics import normalized_mutual_info_score

# -------------------------------------
# --- Parameters
# -------------------------------------

# Segmentation tag ("disc" or "cut")
segm_tag = "cut"

# --- Experiments loop lists (to make several experiments)

# List of inputted text files to explore
input_file_list = ["mix_word1.txt"]
# List of names for the ouputted result files
results_file_name_list = ["results_semisuper5_word1_cut_1.csv"]
# List of label ratios to text
known_label_ratio_list = [0.1]
# List of similarity tag
sim_tag_list = ["w2v"]

# --- Grid search parameters

# Dist options to explore
dist_option_vec = ["minus_log"]
# Exchange matrix options to explore
exch_mat_opt_vec = ["d"]
exch_range_vec = [3, 5, 10, 15]
# Parameter values to explore
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

    # Make results file
    with open(results_file_name, "w") as output_file:
        output_file.write("dist,exch_opt,exch_range,alpha,beta,kappa,nmi\n")

    # -------------------------------------
    # --- Grid search
    # -------------------------------------

    print("Starting Grid search")

    for dist_option in dist_option_vec:

        # Compute the dissimilarity matrix
        d_ext_mat = similarity_to_dissimilarity(sim_ext_mat,
                                                dist_option=dist_option)

        print(f"Dissimilarity matrix computed with {dist_option}")

        for exch_mat_opt in exch_mat_opt_vec:
            for exch_range in exch_range_vec:

                # Compute the exchange and transition matrices
                exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                                   exch_mat_opt=exch_mat_opt,
                                                                   exch_range=exch_range)

                print(f"Exchange matrix computed with {exch_mat_opt} and range {exch_range}")

                for alpha in alpha_vec:
                    for beta in beta_vec:
                        for kappa in kappa_vec:

                            # Compute the matrix
                            result_matrix = segm_function(d_ext_mat=d_ext_mat,
                                                          exch_mat=exch_mat,
                                                          w_mat=w_mat,
                                                          n_groups=n_groups,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          kappa=kappa,
                                                          init_labels=known_labels)


                            # Compute the groups
                            algo_group_value = np.argmax(result_matrix, 1) + 1

                            # Compute nmi score
                            nmi = normalized_mutual_info_score(np.delete(real_group_vec, indices_for_known_label),
                                                               np.delete(algo_group_value, indices_for_known_label))
                            print(f"NMI = {nmi}")

                            # Writing results
                            with open(results_file_name, "a") as output_file:
                                output_file.write(f"{dist_option},{exch_mat_opt},{exch_range},{alpha},{beta},{kappa},"
                                                  f"{nmi}\n")
