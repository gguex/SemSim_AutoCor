from code_python.local_functions import get_all_paths, similarity_to_dissimilarity, type_to_token_matrix_expansion, \
    exchange_and_transition_matrices, cut_segmentation, write_groups_in_html_file, write_membership_mat_in_csv_file
import os
import numpy as np
import csv

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input files list
input_file_list = ["The_WW_of_Oz_nouns.txt", "The_WW_of_Oz_verbs.txt", "Animal_farm_nouns.txt", "Animal_farm_verbs.txt"]
# Similarity tag list
sim_tag_list = ["resnik", "wu-palmer", "leacock-chodorow", "wesim"]

# Distance option
dist_option = "minus_log"
# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive)
exch_mat_opt = "d"
# Exchange matrix range (for uniform) OR time step (for diffusive)
exch_range = 5
# Number of groups
n_groups = 3
# Gamma parameter
gamma = 2
# Beta parameter
beta = 0.4
# Kappa parameter
kappa = 0.7

# -------------------------------------
# --- Computations
# -------------------------------------

# Working path
working_path=os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

for input_file in input_file_list:
    for sim_tag in sim_tag_list:

        # Get the file paths
        text_file_path, typefreq_file_path, sim_file_path, _ = get_all_paths(input_file, sim_tag)

        # Loading the similarity matrix
        sim_mat = np.loadtxt(sim_file_path, delimiter=";")
        # And the corresponding list of types
        with open(typefreq_file_path, 'r') as typefreq_file:
            csv_reader = csv.reader(typefreq_file, delimiter=";")
            type_list = [row[0] for row in csv_reader]

        # Compute the dissimilarity matrix from similarity
        d_mat = similarity_to_dissimilarity(sim_mat,
                                            dist_option=dist_option)

        # Compute the extended version of the matrix
        d_ext_mat, token_list = type_to_token_matrix_expansion(text_file_path, d_mat, type_list)

        # Compute the exchange and transition matrices
        exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                           exch_mat_opt=exch_mat_opt,
                                                           exch_range=exch_range)

        # Compute the membership matrix
        result_matrix = cut_segmentation(d_ext_mat, exch_mat, w_mat,
                                         n_groups=n_groups,
                                         gamma=gamma,
                                         beta=beta,
                                         kappa=kappa)

        # Experiment description
        experiment_description = f"{input_file} | sim_tag: {sim_tag} | dist_option: {dist_option} | " \
                                 f"exch_mat_opt: {exch_mat_opt} | exch_range: {exch_range} | " \
                                 f"n_groups: {n_groups} | gamma: {gamma} | beta: {beta} | kappa: {kappa}"

        # Write html results
        write_groups_in_html_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_cutsegm.html",
                                  token_list, result_matrix, experiment_description)
        # Write csv results
        write_membership_mat_in_csv_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_cutsegm.csv",
                                         token_list, result_matrix)
