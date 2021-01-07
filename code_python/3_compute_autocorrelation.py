from code_python.local_functions import get_all_paths, similarity_to_dissimilarity, type_to_token_matrix_expansion, \
    exchange_and_transition_matrices, autocorrelation_index
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input files list
input_file_list = ["The_WW_of_Oz_pp.txt"]
# Similarity tag list
sim_tag_list = ["jcnb"]

# Distance option
dist_option = "max_minus"
# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive, "r" = ring)
exch_mat_opt = "s"
# Exchange matrix max range
exch_max_range = 50

# -------------------------------------
# --- Computations
# -------------------------------------

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
# Save the windows for autocorrelation range
exch_range_window = list(range(1, exch_max_range + 1))

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

        # Compute autocor vector
        autocor_vec = []
        for exch_range in tqdm(exch_range_window):

            # Compute the exchange and transition matrices
            exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                               exch_mat_opt=exch_mat_opt,
                                                               exch_range=exch_range)
            # Compute the autocorrelation index
            autocor_index = autocorrelation_index(d_ext_mat, exch_mat)
            # Append result to the autocor vector
            autocor_vec.append(autocor_index)

        # Experiment description
        experiment_description = f"{input_file} | sim_tag: {sim_tag} | dist_option: {dist_option} | " \
                                 f"exch_mat_opt: {exch_mat_opt} | exch_max_range: {exch_max_range}"
        # Plot the autocor vector
        plt.figure("Autocorrelation")
        plt.scatter(exch_range_window, autocor_vec)
        plt.plot(exch_range_window, autocor_vec)
        plt.title(experiment_description)
        plt.xlabel("Neighbourhood size r")
        plt.ylabel("Autocorrelation index")
        plt.savefig(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_autocor{exch_max_range}.png")
        plt.close()
