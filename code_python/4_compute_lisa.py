from code_python.local_functions import get_all_paths, similarity_to_dissimilarity, type_to_token_matrix_expansion, \
    exchange_and_transition_matrices, lisa_computation, write_vector_in_html_file
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# -------------------------------------
# --- Parameters
# -------------------------------------

# Input files list
input_file_list = ["Civil_Disobedience_pp.txt",
                   "Flowers_of_the_Farm_pp.txt",
                   "Sidelights_on_relativity_pp.txt",
                   "Prehistoric_Textile_pp.txt"]

# Similarity tag list
sim_tag_list = ["w2v", "glv", "lch", "path", "wup"]

# Distance option
dist_option = "max_minus"
# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive, "r" = ring)
exch_mat_opt = "s"
# Exchange matrix range
exch_range = 50

# -------------------------------------
# --- Computations
# -------------------------------------

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

for input_file in input_file_list:
    for sim_tag in sim_tag_list:

        # Print
        print(f"Computation for {input_file} with {sim_tag}")

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

        # Compute the lisa vector
        lisa_vec = lisa_computation(d_ext_mat, exch_mat, w_mat)

        # Experiment description
        experiment_description = f"{input_file} | sim_tag: {sim_tag} | dist_option: {dist_option} | " \
                                 f"exch_mat_opt: {exch_mat_opt} | exch_range: {exch_range}"

        # Write the html file from lisa vector
        write_vector_in_html_file(f"{base_path}/results/4_lisa{exch_range}_{input_file[:-4]}_{sim_tag}.html",
                                  token_list, lisa_vec, experiment_description)

        # Write the plot
        plt.figure("Lisa index")
        plt.plot(list(range(1, len(token_list) + 1)), lisa_vec, linewidth=0.1)
        plt.title(f"Lisa on {input_file[:-4]} with {sim_tag}")
        plt.xlabel("Token")
        plt.ylabel("Lisa index")
        plt.savefig(f"{base_path}/results/4_lisa{exch_range}_{input_file[:-4]}_{sim_tag}.png")
        plt.close()
