from local_functions import get_all_paths, similarity_to_dissimilarity, type_to_token_matrix_expansion, \
    exchange_and_transition_matrices, autocorrelation_index
import os
import numpy as np
import csv
import multiprocessing as mp
from miniutils import parallel_progbar

# -------------------------------------
# --- Parameters
# -------------------------------------

# Similarity tag list
sim_tag_list = ["glv", "w2v", "lch", "path", "wup"]
# Input files list
input_file_list = ["Civil_Disobedience_pp.txt",
                   "Flowers_of_the_Farm_pp.txt",
                   "Sidelights_on_relativity_pp.txt",
                   "Prehistoric_Textile_pp.txt"]
# Distance option
dist_option = "max_minus"
# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive, "r" = ring)
exch_mat_opt = "u"
# Exchange matrix max range
exch_max_range = 50
# Number of cpus to use
n_cpu = mp.cpu_count()

# -------------------------------------
# --- Computations
# -------------------------------------

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
# Save the windows for autocorrelation range
exch_range_window = list(range(1, exch_max_range + 1))

for sim_tag in sim_tag_list:

    # Output file name
    output_file_name = f"{base_path}/results/3_autocor_results/3.1_autocor{exch_max_range}_{sim_tag}.csv"
    # Write header
    with open(output_file_name, "w") as output_file:
        output_file.write("csv_file_name, " + f"{list(range(1, 1 + exch_max_range))}"[1:-1] + "\n")

    for input_file in input_file_list:

        # Write file name
        with open(output_file_name, "a") as output_file:
            output_file.write(input_file[:-7])
        # Print
        print(f"Autocorrelation for {input_file} with similarity {sim_tag}")

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

        # the z_autocor function
        def z_autocor(exch_range):
            # Compute the exchange and transition matrices
            exch_mat, w_mat = exchange_and_transition_matrices(len(token_list),
                                                               exch_mat_opt=exch_mat_opt,
                                                               exch_range=exch_range)
            # Compute the autocorrelation index
            autocor_index, theoretical_mean, theoretical_var = autocorrelation_index(d_ext_mat, exch_mat, w_mat)

            # Z score for autocor
            z_ac = (autocor_index - theoretical_mean) / np.sqrt(theoretical_var)

            # Return
            return z_ac

        # Run on multiprocess
        z_autocor_vec = parallel_progbar(z_autocor, exch_range_window, nprocs=n_cpu)

        with open(output_file_name, "a") as output_file:
            for z_val in z_autocor_vec:
                output_file.write(f", {z_val}")
            output_file.write("\n")