import os
from code_python.local_functions import sim_to_dissim_matrix, exchange_and_transition_matrices, \
    discontinuity_segmentation

# --- Parameters --- #

# Input files list
input_files = ("The_WW_of_Oz_nouns.txt", "The_WW_of_Oz_verbs.txt", "Animal_farm_nouns.txt", "Animal_farm_verbs.txt")
# Similarity tag list
sim_tags = ("resnik", "wu-palmer", "leacock-chodorow", "wesim")

# Exchange matrix option ("s" = standard, "u" = uniform, "d" = diffusive)
exch_mat_opt = "d"
# Exchange matrix range (for uniform) OR time step (for diffusive)
exch_range = 5
# Number of groups
n_groups = 5
# Alpha parameter
alpha = 5
# Beta parameter
beta = 50
# Kappa parameter
kappa = 0.8

# --- Defining paths --- #

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor/"

# --- Computing results --- #

for input_file in input_files:
    for sim_tag in sim_tags:
        d_ext_mat = sim_to_dissim_matrix(input_file, sim_tag)
        exch_mat, w_mat = exchange_and_transition_matrices(input_file, sim_tag, exch_mat_opt, exch_range)
        w_mat = discontinuity_segmentation(exch_mat, w_mat, n_groups, alpha, beta, kappa)

