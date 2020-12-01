import os
from code_python.local_functions import sim_to_dissim, exchange_and_transition_matrices, \
    cut_segmentation, write_groups_in_html_file, write_membership_mat_in_csv_file

# --- Parameters --- #

# Input files list
input_files = ["The_WW_of_Oz_nouns.txt", "The_WW_of_Oz_verbs.txt", "Animal_farm_nouns.txt", "Animal_farm_verbs.txt"]
# Similarity tag list
sim_tags = ["resnik", "wu-palmer", "leacock-chodorow", "wesim"]

# Exchange matrix option ("u" = uniform, "d" = diffusive)
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


# --- Defining paths --- #

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"

# --- Computing results --- #

for input_file in input_files:
    for sim_tag in sim_tags:

        # Compute values
        d_ext_mat, token_list = sim_to_dissim(input_file, sim_tag)
        exch_mat, w_mat = exchange_and_transition_matrices(len(token_list), exch_mat_opt, exch_range)
        result_matrix = cut_segmentation(d_ext_mat, exch_mat, w_mat, n_groups, gamma, beta, kappa)

        # Write html results
        write_groups_in_html_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_cutsegm.html",
                                  token_list, result_matrix)
        # Write csv results
        write_membership_mat_in_csv_file(f"{base_path}/results/{input_file[:-4]}_{sim_tag}_cutsegm.csv",
                                         token_list, result_matrix)
