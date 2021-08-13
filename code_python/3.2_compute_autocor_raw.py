from local_functions import get_all_paths, similarity_to_dissimilarity
import os
import numpy as np
import multiprocessing as mp
from os.path import expanduser
import nltk
from gensim.models import KeyedVectors
import re
from tqdm import tqdm

# -------------------------------------
# --- Parameters
# -------------------------------------

# Similarity tag list
sim_tag_list = ["w2v"]
# Input files list
# input_file_list = ["gutenberg/ShakespeareWilliam_Othello.txt"]
input_file_list = ["gutenberg/CarrollLewis_AliceAdventures.txt",
                   "gutenberg/CarrollLewis_SylvieandBruno.txt",
                   "gutenberg/CarrollLewis_TheHuntingoftheSnark.txt",
                   "gutenberg/CarrollLewis_ThroughtheLookingGlass.txt",
                   "gutenberg/ShakespeareWilliam_Hamlet.txt",
                   "gutenberg/ShakespeareWilliam_Macbeth.txt",
                   "gutenberg/ShakespeareWilliam_Othello.txt",
                   "gutenberg/ShakespeareWilliam_RomeoandJuliette.txt",
                   "gutenberg/WellsHG_DoctorMoreau.txt",
                   "gutenberg/WellsHG_TheFirstMenintheMoon.txt",
                   "gutenberg/WellsHG_TheInvisibleMan.txt",
                   "gutenberg/WellsHG_TheTimeMachine.txt",
                   "gutenberg/WildeOscar_AnIdealHusband.txt",
                   "gutenberg/WildeOscar_TheCantervilleGhost.txt",
                   "gutenberg/WildeOscar_TheImportanceofBeingEarnes.txt",
                   "gutenberg/WildeOscar_ThePictureofDorianGray.txt",
                   "CarrollLewis_AliceAdventures_pp.txt",
                   "CarrollLewis_SylvieandBruno_pp.txt",
                   "CarrollLewis_TheHuntingoftheSnark_pp.txt",
                   "CarrollLewis_ThroughtheLookingGlass_pp.txt",
                   "ShakespeareWilliam_Hamlet_pp.txt",
                   "ShakespeareWilliam_Macbeth_pp.txt",
                   "ShakespeareWilliam_Othello_pp.txt",
                   "ShakespeareWilliam_RomeoandJuliette_pp.txt",
                   "WellsHG_DoctorMoreau_pp.txt",
                   "WellsHG_TheFirstMenintheMoon_pp.txt",
                   "WellsHG_TheInvisibleMan_pp.txt",
                   "WellsHG_TheTimeMachine_pp.txt",
                   "WildeOscar_AnIdealHusband_pp.txt",
                   "WildeOscar_TheCantervilleGhost_pp.txt",
                   "WildeOscar_TheImportanceofBeingEarnes_pp.txt",
                   "WildeOscar_ThePictureofDorianGray_pp.txt"]

# Distance option
dist_option = "max_minus"
# Exchange matrix max range
exch_range_window = list(range(1, 50 + 1))
# Number of cpus to use
n_cpu = mp.cpu_count() - 2
# Max token
max_token = 8000

# -------------------------------------
# --- Computations
# -------------------------------------

# Working path
working_path = os.getcwd()
# Getting the SemSim_AutoCor folder, if above
base_path = str.split(working_path, "SemSim_AutoCor")[0] + "SemSim_AutoCor"
# Save the windows for autocorrelation range

# Getting home path
home = expanduser("~")

for sim_tag in sim_tag_list:

    # Get the wv path
    if sim_tag == "w2v":
        word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/enwiki.model"
    elif sim_tag == "glv":
        word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/glove42B300d.model"
    else:
        word_vector_path = f"{home}/Documents/data/pretrained_word_vectors/en_fasttext.model"

    # Loading wordvector models
    wv_model = KeyedVectors.load(word_vector_path)
    # Vocabulary of word vectors
    vocab_wv = set(wv_model.vocab.keys())

    # Output file name
    output_file_name = f"{base_path}/results/3.2_autocor{max(exch_range_window)}_{sim_tag}.csv"
    # Write header
    with open(output_file_name, "w") as output_file:
        output_file.write("csv_file_name, " + f"{exch_range_window}"[1:-1] + "\n")

    for input_file in input_file_list:

        # Write file name
        with open(output_file_name, "a") as output_file:
            output_file.write(re.sub(r'[^a-zA-Z ]', '', input_file[:-7]).lower())
        # Print
        print(f"Autocorrelation for {input_file} with similarity {sim_tag}")

        # Get the file paths
        text_file_path, _, _, _ = get_all_paths(input_file, sim_tag)

        # Opening the file
        with open(text_file_path, "r") as text_file:
            text_string = text_file.read()
            text_string = re.sub(r'[^a-zA-Z ]', '', text_string).lower()
        # Split by tokens
        token_list = nltk.word_tokenize(text_string)
        # Reduce it if too long
        if len(token_list) > max_token:
            nb_excess_token = len(token_list) - max_token
            token_list = token_list[int(nb_excess_token / 2):int(nb_excess_token / 2 + max_token + 1)]
        # Vocabulary of text
        vocab_text = set(token_list)

        # The common vocabulary
        vocab_common = list(vocab_wv & vocab_text)
        # Number of type
        n_type = len(vocab_common)

        # Reducing token_list to existing one
        existing_pos_list = [token in vocab_common for token in token_list]
        existing_token_list = np.array(token_list)[existing_pos_list]
        # Number of token
        n_token = len(existing_token_list)

        # Compute the n_type x n_token presence matrix and the sim matrix
        pres_mat = np.empty([0, n_token])
        sim_mat = np.eye(n_type) / 2
        for ind_type_1, type_1 in tqdm(enumerate(vocab_common), total=len(vocab_common)):
            # The freq
            pres_mat = np.append(pres_mat, [[token == type_1 for token in existing_token_list]], axis=0)
            for ind_type_2 in range(ind_type_1 + 1, n_type):
                sim_mat[ind_type_1, ind_type_2] = \
                    wv_model.similarity(type_1, vocab_common[ind_type_2])
        sim_mat = sim_mat + sim_mat.T

        # Compute the dissimilarity matrix from similarity
        d_mat = similarity_to_dissimilarity(sim_mat, dist_option=dist_option)

        # Compute the fixed quantites
        f_vec = np.ones(n_token) / n_token
        range_help_mat = np.abs(np.add.outer(np.arange(n_token), -np.arange(n_token)))


        # the z_autocor func
        def z_autocor(exch_range):

            # Compute the exchange matrix
            adj_mat = range_help_mat <= exch_range
            np.fill_diagonal(adj_mat, 0)
            g_vec = np.sum(adj_mat, axis=1) / np.sum(adj_mat)
            k_vec = f_vec / g_vec
            b_mat = np.minimum(np.outer(k_vec, np.ones(n_token)), np.outer(np.ones(n_token), k_vec)) * adj_mat \
                    / np.sum(adj_mat)
            exch_mat = np.diag(f_vec) - np.diag(np.sum(b_mat, axis=1)) + b_mat

            # Compute the W mat
            w_mat = (exch_mat / f_vec).T

            # Reduce the exchange matrix and compute frequency by type
            exch_mat_red = pres_mat @ exch_mat @ pres_mat.T
            pi_vec = np.sum(exch_mat_red, axis=0)

            # Compute the global/local var and autocor index
            global_var = 0.5 * np.sum((d_mat * pi_vec).T * pi_vec)
            local_var = 0.5 * np.sum(exch_mat_red * d_mat)
            autocor_index = 1 - local_var / global_var

            # Compute the theoretical expected value
            trace_w_mat = np.trace(w_mat)
            theoretical_mean = (trace_w_mat - 1) / (n_token - 1)
            # Compute the theoretical
            theoretical_var = 2 * (np.trace(w_mat @ w_mat) - 1 - (trace_w_mat - 1) ** 2 / (n_token - 1)) \
                              / (n_token ** 2 - 1)

            # Z score for autocor
            z_ac = (autocor_index - theoretical_mean) / np.sqrt(theoretical_var)

            # Return
            return z_ac


        # Run on multiprocess
        with mp.Pool(n_cpu) as mypool:
            z_autocor_vec = list(tqdm(mypool.imap(z_autocor, exch_range_window), total=len(exch_range_window)))

        # z_autocor_vec = []
        # for exch_range in tqdm(exch_range_window):
        #
        #     # Compute the exchange matrix
        #     adj_mat = range_help_mat <= exch_range
        #     np.fill_diagonal(adj_mat, 0)
        #     g_vec = np.sum(adj_mat, axis=1) / np.sum(adj_mat)
        #     k_vec = f_vec / g_vec
        #     b_mat = np.minimum(np.outer(k_vec, np.ones(n_token)), np.outer(np.ones(n_token), k_vec)) * adj_mat \
        #             / np.sum(adj_mat)
        #     exch_mat = np.diag(f_vec) - np.diag(np.sum(b_mat, axis=1)) + b_mat
        #
        #     # Compute the W mat
        #     w_mat = (exch_mat / f_vec).T
        #
        #     # Reduce the exchange matrix and compute frequency by type
        #     exch_mat_red = pres_mat @ exch_mat @ pres_mat.T
        #     pi_vec = np.sum(exch_mat_red, axis=0)
        #
        #     # Compute the global/local var and autocor index
        #     global_var = 0.5 * np.sum((d_mat * pi_vec).T * pi_vec)
        #     local_var = 0.5 * np.sum(exch_mat_red * d_mat)
        #     autocor_index = 1 - local_var / global_var
        #
        #     # Compute the theoretical expected value
        #     trace_w_mat = np.trace(w_mat)
        #     theoretical_mean = (trace_w_mat - 1) / (n_token - 1)
        #     # Compute the theoretical
        #     theoretical_var = 2 * (np.trace(w_mat @ w_mat) - 1 - (trace_w_mat - 1) ** 2 / (n_token - 1)) \
        #                       / (n_token ** 2 - 1)
        #
        #     # Z score for autocor
        #     z_autocor_vec.append((autocor_index - theoretical_mean) / np.sqrt(theoretical_var))

        with open(output_file_name, "a") as output_file:
            for z_val in z_autocor_vec:
                output_file.write(f", {z_val}")
            output_file.write("\n")
