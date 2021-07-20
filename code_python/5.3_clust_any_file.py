from local_functions import get_all_paths, similarity_to_dissimilarity, type_to_token_matrix_expansion, \
    exchange_and_transition_matrices, discontinuity_clustering, cut_clustering, write_groups_in_html_file, \
    write_membership_mat_in_csv_file
import os
import numpy as np
import csv

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input files list
input_file_list = ["The_WW_of_Oz_pp.txt"]
# Similarity tag list
sim_tag_list = ["w2v"]

# Distance option
dist_option = "max_minus"
# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive)
exch_mat_opt = "u"
# Exchange matrix range (for uniform) OR time step (for diffusive)
exch_range = 10
# Number of groups
n_group = 5
# Alpha parameter
alpha = 10
# Beta parameter
beta = 10
# Kappa parameter
kappa = 0.8
# Clustering method tag ("disc" or "cut")"
clust_tag = "cut"

# -------------------------------------
# --- Computations
# -------------------------------------

# Selection of the clustering function
if clust_tag == "disc":
    segm_function = discontinuity_clustering
else:
    segm_function = cut_clustering

# Working path
working_path = os.getcwd()
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
        d_ext_mat, token_list, _ = type_to_token_matrix_expansion(text_file_path, d_mat, type_list)

        # Compute the exchange and transition matrices
        exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                           exch_mat_opt=exch_mat_opt,
                                                           exch_range=exch_range)

        # Compute the membership matrix
        result_matrix = segm_function(d_ext_mat, exch_mat, w_mat,
                                      n_group=n_group,
                                      alpha=alpha,
                                      beta=beta,
                                      kappa=kappa)

        # Experiment description
        experiment_description = f"{input_file} | segm_tag: {clust_tag} | sim_tag: {sim_tag} | " \
                                 f"dist_option: {dist_option} | exch_mat_opt: {exch_mat_opt} | " \
                                 f"exch_range: {exch_range} | n_groups: {n_group} | " \
                                 f"alpha: {alpha} | beta: {beta} | kappa: {kappa}"

        # Write html results
        write_groups_in_html_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_{clust_tag}.html",
                                  token_list, result_matrix, experiment_description)
        # Write csv results
        write_membership_mat_in_csv_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_{clust_tag}.csv",
                                         token_list, result_matrix)
